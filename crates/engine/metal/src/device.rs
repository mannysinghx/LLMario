//! The Metal FFI layer: device, command queue, runtime-compiled kernel library, pipeline states,
//! buffers and residency sets, wrapped in a small safe API. Everything `unsafe` in this crate
//! lives here, with the invariant each call relies on stated next to it.
//!
//! Threading: Metal devices, queues, libraries, pipeline states and buffers are documented as
//! thread-safe; command buffers and encoders are not and never outlive one `forward` call on the
//! inference thread.

use crate::{DeviceInfo, MetalError, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::sel;
use objc2_foundation::{NSProcessInfo, NSString};
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType, MTLGPUFamily, MTLLanguageVersion,
    MTLLibrary, MTLResidencySet, MTLResidencySetDescriptor, MTLResourceOptions, MTLResourceUsage,
    MTLSize,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

/// The embedded kernel source (compiled once per process by the OS Metal compiler).
pub const SHADER_SOURCE: &str = concat!(
    include_str!("shaders/common.metal"),
    "\n",
    include_str!("shaders/gemv.metal"),
    "\n",
    include_str!("shaders/gemm.metal"),
    "\n",
    include_str!("shaders/misc.metal"),
    "\n",
    include_str!("shaders/attn_prefill.metal"),
    "\n",
    include_str!("shaders/moe.metal"),
);

/// Every kernel the backend binds, by `host_name`.
pub const KERNELS: &[&str] = &[
    "embed_f32",
    "embed_f16",
    "embed_q4_0",
    "embed_q8_0",
    "embed_q4_k",
    "embed_q5_k",
    "embed_q6_k",
    "gemv_f32",
    "gemv_f16",
    "gemv_q4_0",
    "gemv_q8_0",
    "gemv_q4_k",
    "gemv_q5_k",
    "gemv_q6_k",
    "gemv_acc_f32",
    "gemv_acc_f16",
    "gemv_acc_q4_0",
    "gemv_acc_q8_0",
    "gemv_acc_q4_k",
    "gemv_acc_q5_k",
    "gemv_acc_q6_k",
    "gemv_glu_f32",
    "gemv_glu_f16",
    "gemv_glu_q4_0",
    "gemv_glu_q8_0",
    "gemv_glu_q4_k",
    "gemv_glu_q5_k",
    "gemv_glu_q6_k",
    "gemv_geglu_f32",
    "gemv_geglu_f16",
    "gemv_geglu_q4_0",
    "gemv_geglu_q8_0",
    "gemv_geglu_q4_k",
    "gemv_geglu_q5_k",
    "gemv_geglu_q6_k",
    "gemm_f32",
    "gemm_f16",
    "gemm_q4_0",
    "gemm_q8_0",
    "gemm_q4_k",
    "gemm_q5_k",
    "gemm_q6_k",
    "rms_norm",
    "qk_rope_kv_f16",
    "qk_rope_kv_q8_0",
    "attn_vec_hd32_f16",
    "attn_vec_hd64_f16",
    "attn_vec_hd128_f16",
    "attn_vec_hd256_f16",
    "attn_vec_hd512_f16",
    "attn_vec_generic_f16",
    "attn_vec_hd32_q8_0",
    "attn_vec_hd64_q8_0",
    "attn_vec_hd128_q8_0",
    "attn_vec_hd256_q8_0",
    "attn_vec_hd512_q8_0",
    "attn_vec_generic_q8_0",
    "attn_prefill_hd64_f16",
    "attn_prefill_hd128_f16",
    "attn_prefill_hd64_q8_0",
    "attn_prefill_hd128_q8_0",
    "attn_reduce",
    "swiglu",
    "geglu",
    "scale_inplace",
    "softcap",
    "rms_norm_add",
    "add",
    "add_bias",
    "gemv_id_f32",
    "gemv_id_f16",
    "gemv_id_q4_0",
    "gemv_id_q8_0",
    "gemv_id_q4_k",
    "gemv_id_q5_k",
    "gemv_id_q6_k",
    "gemv_glu_id_f32",
    "gemv_glu_id_f16",
    "gemv_glu_id_q4_0",
    "gemv_glu_id_q8_0",
    "gemv_glu_id_q4_k",
    "gemv_glu_id_q5_k",
    "gemv_glu_id_q6_k",
    "gemm_id_f32",
    "gemm_id_f16",
    "gemm_id_q4_0",
    "gemm_id_q8_0",
    "gemm_id_q4_k",
    "gemm_id_q5_k",
    "gemm_id_q6_k",
    "moe_route",
    "moe_group",
    "moe_combine",
];

type Device = ProtocolObject<dyn MTLDevice>;
type Queue = ProtocolObject<dyn MTLCommandQueue>;
type Pso = ProtocolObject<dyn MTLComputePipelineState>;
type RawBuffer = ProtocolObject<dyn MTLBuffer>;

/// A shared-storage Metal buffer (either allocated by Metal or a no-copy view over a mapping).
pub struct Buf {
    raw: Retained<RawBuffer>,
    len: usize,
}

// SAFETY: `MTLBuffer` objects are thread-safe per Apple's Metal threading documentation; the
// binding omits the auto-traits only because the protocol is not annotated. The backend moves
// buffers to the inference thread and never shares them across threads concurrently.
unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

impl Buf {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// The buffer's GPU virtual address (Metal 3), for tables that kernels read pointers from.
    pub fn gpu_address(&self) -> u64 {
        self.raw.gpuAddress()
    }
    /// The CPU-visible contents (shared storage mode). The caller must not read or write while a
    /// command buffer that uses this buffer is executing; the backend waits on every forward.
    pub fn contents(&self) -> *mut u8 {
        self.raw.contents().as_ptr() as *mut u8
    }
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `contents` is valid for `len` bytes for the buffer's lifetime (shared storage).
        unsafe { std::slice::from_raw_parts(self.contents(), self.len) }
    }
    /// Copy `src` into the buffer at byte offset `off`.
    pub fn write_bytes(&self, off: usize, src: &[u8]) {
        assert!(off + src.len() <= self.len, "write past buffer end");
        // SAFETY: bounds checked above; shared-mode contents are CPU-writable and no GPU work is in
        // flight (the backend waits for completion before touching buffers).
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.contents().add(off), src.len()) }
    }
    pub fn write_f32(&self, off_bytes: usize, src: &[f32]) {
        // SAFETY: f32 slices are plain bytes.
        let bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) };
        self.write_bytes(off_bytes, bytes);
    }
    /// Read `n` f32 values starting at byte offset `off`.
    pub fn read_f32(&self, off_bytes: usize, dst: &mut [f32]) {
        assert!(
            off_bytes + dst.len() * 4 <= self.len,
            "read past buffer end"
        );
        // SAFETY: bounds checked; shared-mode contents are CPU-readable after the command buffer
        // completed; `read_unaligned`-free because the offsets we use are 4-byte aligned.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.contents().add(off_bytes) as *const f32,
                dst.as_mut_ptr(),
                dst.len(),
            )
        }
    }
}

/// One Metal device with its queue and compiled pipelines, shared by every backend in the process.
pub struct Gpu {
    device: Retained<Device>,
    queue: Retained<Queue>,
    pipelines: HashMap<&'static str, Retained<Pso>>,
    /// Residency sets kept alive for the queue (macOS 15+; empty when unsupported).
    /// Residency sets kept alive for the queue, by owner id (several backends can share the
    /// process-wide device, and they are created and dropped in any order).
    residency: Mutex<Vec<(u64, ResidencySet)>>,
    pub info: DeviceInfo,
    pub max_buffer_length: usize,
    pub page_size: usize,
}

// SAFETY: devices, queues and pipeline states are thread-safe Metal objects; `residency` is
// behind a mutex. See the module doc.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

pub struct ResidencySet {
    raw: Retained<ProtocolObject<dyn MTLResidencySet>>,
}

static NEXT_RESIDENCY_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A fresh owner id for [`Gpu::make_resident`] / [`Gpu::end_residency`].
pub fn residency_owner() -> u64 {
    NEXT_RESIDENCY_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn ns_err(e: &objc2_foundation::NSError) -> String {
    e.localizedDescription().to_string()
}

/// Create the system default device if it is usable for this backend.
fn default_device() -> Option<Retained<Device>> {
    let dev = MTLCreateSystemDefaultDevice()?;
    let name = dev.name().to_string();
    // Apple7 (M1) is the first family with simdgroup matrix multiply and simd reductions.
    if !dev.supportsFamily(MTLGPUFamily::Apple7) {
        tracing::info!(device = %name, "Metal device lacks GPU family Apple7; backend disabled");
        return None;
    }
    // Virtualised GPUs (CI runners, VMs) report the family but run the kernels on a software
    // path with incomplete feature support; the CPU backend is the right choice there.
    if name.contains("Paravirtual") {
        tracing::info!(device = %name, "paravirtual Metal device; backend disabled");
        return None;
    }
    Some(dev)
}

pub fn is_available() -> bool {
    default_device().is_some()
}

fn family_name(dev: &Device) -> String {
    let fams = [
        (MTLGPUFamily::Apple10, "apple10"),
        (MTLGPUFamily::Apple9, "apple9"),
        (MTLGPUFamily::Apple8, "apple8"),
        (MTLGPUFamily::Apple7, "apple7"),
    ];
    let mut out = fams
        .iter()
        .find(|(f, _)| dev.supportsFamily(*f))
        .map(|(_, n)| n.to_string())
        .unwrap_or_else(|| "unknown".into());
    if dev.supportsFamily(MTLGPUFamily::Metal4) {
        out.push_str("+metal4");
    } else if dev.supportsFamily(MTLGPUFamily::Metal3) {
        out.push_str("+metal3");
    }
    out
}

pub fn device_info() -> Option<DeviceInfo> {
    let dev = default_device()?;
    Some(info_of(&dev))
}

fn info_of(dev: &Device) -> DeviceInfo {
    DeviceInfo {
        name: dev.name().to_string(),
        recommended_max_working_set: dev.recommendedMaxWorkingSetSize(),
        has_unified_memory: dev.hasUnifiedMemory(),
        family: family_name(dev),
        residency_sets: residency_supported(dev),
    }
}

fn residency_supported(dev: &Device) -> bool {
    use objc2::runtime::NSObjectProtocol;
    let os = NSProcessInfo::processInfo().operatingSystemVersion();
    os.majorVersion >= 15 && dev.respondsToSelector(sel!(newResidencySetWithDescriptor:error:))
}

static GPU: OnceLock<std::result::Result<Arc<Gpu>, String>> = OnceLock::new();

impl Gpu {
    /// The process-wide device, compiled on first use.
    pub fn get() -> Result<Arc<Gpu>> {
        GPU.get_or_init(|| Gpu::create().map(Arc::new).map_err(|e| e.to_string()))
            .clone()
            .map_err(MetalError::Device)
    }

    fn create() -> Result<Gpu> {
        let device =
            default_device().ok_or_else(|| MetalError::Device("no usable Metal device".into()))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| MetalError::Device("newCommandQueue failed".into()))?;
        let t0 = std::time::Instant::now();
        let opts = MTLCompileOptions::new();
        // Metal 3.0 is what macOS 13+ ships; set explicitly so the language level is never
        // silently inherited from the SDK (the llama.cpp "silently inert tensor path" lesson).
        opts.setLanguageVersion(MTLLanguageVersion::Version3_0);
        let src = NSString::from_str(SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&src, Some(&opts))
            .map_err(|e| MetalError::Compile(ns_err(&e)))?;
        let mut pipelines = HashMap::new();
        for &name in KERNELS {
            let func = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| MetalError::Compile(format!("kernel `{name}` not found")))?;
            let pso = device
                .newComputePipelineStateWithFunction_error(&func)
                .map_err(|e| MetalError::Compile(format!("pipeline `{name}`: {}", ns_err(&e))))?;
            pipelines.insert(name, pso);
        }
        let info = info_of(&device);
        tracing::info!(
            device = %info.name,
            family = %info.family,
            kernels = pipelines.len(),
            secs = t0.elapsed().as_secs_f32(),
            "Metal kernels compiled"
        );
        // SAFETY: sysconf is always safe to call.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize;
        Ok(Gpu {
            max_buffer_length: device.maxBufferLength(),
            page_size,
            device,
            queue,
            pipelines,
            residency: Mutex::new(Vec::new()),
            info,
        })
    }

    pub fn pipeline(&self, name: &str) -> &Pso {
        self.pipelines
            .get(name)
            .unwrap_or_else(|| panic!("kernel `{name}` not in KERNELS"))
    }

    /// Bytes currently allocated by Metal on this device (ledger sample).
    pub fn current_allocated_size(&self) -> u64 {
        self.device.currentAllocatedSize() as u64
    }

    /// Allocate a zero-initialised shared-storage buffer.
    pub fn alloc(&self, len: usize) -> Result<Buf> {
        let len = len.max(16);
        let raw = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::Alloc(len))?;
        let b = Buf { raw, len };
        // SAFETY: fresh shared buffer of `len` bytes, no GPU work in flight.
        unsafe { std::ptr::write_bytes(b.contents(), 0, len) };
        Ok(b)
    }

    /// Wrap `len` bytes at `ptr` (page-aligned, `len` a page multiple, within one mapping) as a
    /// shared-storage buffer without copying. The mapping must outlive the returned buffer.
    ///
    /// # Safety
    /// `ptr..ptr+len` must stay mapped and readable for as long as the buffer exists.
    pub unsafe fn wrap_no_copy(&self, ptr: *const u8, len: usize) -> Result<Buf> {
        debug_assert_eq!(ptr as usize % self.page_size, 0);
        debug_assert_eq!(len % self.page_size, 0);
        let nn = NonNull::new(ptr as *mut c_void)
            .ok_or_else(|| MetalError::Device("null mapping".into()))?;
        // SAFETY: caller guarantees the range is a live, page-aligned mapping; no deallocator is
        // passed, so Metal never frees memory it does not own (ggml-metal does the same).
        let raw = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    nn,
                    len,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or(MetalError::Alloc(len))?;
        Ok(Buf { raw, len })
    }

    /// Put `bufs` in a residency set attached to the queue (macOS 15+); no-op elsewhere.
    pub fn make_resident(&self, owner: u64, label: &str, bufs: &[&Buf]) -> bool {
        if !self.info.residency_sets || bufs.is_empty() {
            return false;
        }
        let desc = MTLResidencySetDescriptor::new();
        desc.setLabel(Some(&NSString::from_str(label)));
        // SAFETY: a plain capacity hint.
        unsafe { desc.setInitialCapacity(bufs.len()) };
        let set = match self.device.newResidencySetWithDescriptor_error(&desc) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %ns_err(&e), "residency set creation failed");
                return false;
            }
        };
        for b in bufs {
            set.addAllocation(ProtocolObject::from_ref(&*b.raw));
        }
        set.commit();
        set.requestResidency();
        self.queue.addResidencySet(&set);
        self.residency
            .lock()
            .unwrap()
            .push((owner, ResidencySet { raw: set }));
        true
    }

    /// A residency set whose members change over time (the paged KV blocks, which kernels reach
    /// only through address tables). `None` where residency sets are unsupported; then the
    /// backend declares the buffers on each command encoder instead (`Cmd::use_indirect`).
    pub fn dynamic_residency(&self, label: &str) -> Option<DynResidency> {
        if !self.info.residency_sets {
            return None;
        }
        let desc = MTLResidencySetDescriptor::new();
        desc.setLabel(Some(&NSString::from_str(label)));
        let set = match self.device.newResidencySetWithDescriptor_error(&desc) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %ns_err(&e), "residency set creation failed");
                return None;
            }
        };
        set.commit();
        set.requestResidency();
        self.queue.addResidencySet(&set);
        Some(DynResidency {
            raw: set,
            queue: self.queue.clone(),
            dirty: false,
        })
    }

    /// Release the residency sets `owner` created (called on drop; the memory stays mapped).
    pub fn end_residency(&self, owner: u64) {
        let mut sets = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        sets.retain(|(o, s)| {
            if *o == owner {
                self.queue.removeResidencySet(&s.raw);
                s.raw.endResidency();
                false
            } else {
                true
            }
        });
    }

    pub fn residency_count(&self) -> usize {
        self.residency
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Begin one command buffer with one compute encoder (serial dispatch: every kernel sees the
    /// results of the kernels encoded before it).
    pub fn begin(&self) -> Result<Cmd<'_>> {
        Ok(Cmd {
            gpu: self,
            open: RefCell::new(None),
            profile: None,
            indirect: RefCell::new(Vec::new()),
        })
    }

    /// Like [`Gpu::begin`], but every dispatch runs in its own command buffer and its GPU time
    /// is recorded (for the kernel profile; much slower than a normal forward).
    pub fn begin_profiled(&self) -> Result<Cmd<'_>> {
        Ok(Cmd {
            gpu: self,
            open: RefCell::new(None),
            profile: Some(RefCell::new(Vec::new())),
            indirect: RefCell::new(Vec::new()),
        })
    }

    fn open_cb(&self) -> Result<OpenCmd> {
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| MetalError::Device("commandBuffer failed".into()))?;
        // Concurrent dispatch: kernels encoded between two `barrier` calls may overlap; the
        // backend places a barrier before every kernel that reads a previous kernel's output.
        let enc = cb
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or_else(|| MetalError::Device("computeCommandEncoder failed".into()))?;
        Ok(OpenCmd { cb, enc })
    }
}

struct OpenCmd {
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

impl OpenCmd {
    /// End encoding, commit, wait; returns the GPU execution time in seconds.
    fn finish(self) -> Result<f64> {
        self.enc.endEncoding();
        self.cb.commit();
        self.cb.waitUntilCompleted();
        if self.cb.status() == MTLCommandBufferStatus::Error {
            let msg = self.cb.error().map(|e| ns_err(&e)).unwrap_or_default();
            return Err(MetalError::Gpu(msg));
        }
        Ok(self.cb.GPUEndTime() - self.cb.GPUStartTime())
    }
}

/// See [`Gpu::dynamic_residency`]. Changes take effect at the next [`DynResidency::commit`].
pub struct DynResidency {
    raw: Retained<ProtocolObject<dyn MTLResidencySet>>,
    queue: Retained<Queue>,
    dirty: bool,
}

// SAFETY: residency sets are thread-safe Metal objects; the backend uses this from its own
// thread only.
unsafe impl Send for DynResidency {}
unsafe impl Sync for DynResidency {}

impl DynResidency {
    pub fn add(&mut self, b: &Buf) {
        self.raw.addAllocation(ProtocolObject::from_ref(&*b.raw));
        self.dirty = true;
    }
    pub fn remove(&mut self, b: &Buf) {
        self.raw.removeAllocation(ProtocolObject::from_ref(&*b.raw));
        self.dirty = true;
    }
    /// Apply pending additions and removals (call before encoding work that uses them).
    pub fn commit(&mut self) {
        if self.dirty {
            self.raw.commit();
            self.raw.requestResidency();
            self.dirty = false;
        }
    }
}

impl Drop for DynResidency {
    fn drop(&mut self) {
        self.queue.removeResidencySet(&self.raw);
        self.raw.endResidency();
    }
}

/// An open command buffer. Kernels are encoded back to back; `finish` commits and waits.
pub struct Cmd<'g> {
    gpu: &'g Gpu,
    open: RefCell<Option<OpenCmd>>,
    /// Per-dispatch (kernel, GPU seconds) when profiling.
    profile: Option<RefCell<Vec<(&'static str, f64)>>>,
    /// Buffers kernels reach only through address tables, declared on every encoder this command
    /// opens (needed where residency sets are unavailable).
    indirect: RefCell<Vec<Retained<RawBuffer>>>,
}

impl Cmd<'_> {
    /// Declare buffers that kernels read or write only through address tables. Applies to the
    /// current encoder and to every encoder opened later by this command.
    pub fn use_indirect(&self, bufs: &[&Buf]) {
        let mut ind = self.indirect.borrow_mut();
        ind.clear();
        ind.extend(bufs.iter().map(|b| b.raw.clone()));
        if let Some(o) = self.open.borrow().as_ref() {
            for b in ind.iter() {
                o.enc.useResource_usage(
                    ProtocolObject::from_ref(&**b),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
            }
        }
    }

    fn ensure_open(&self) -> Result<()> {
        let mut open = self.open.borrow_mut();
        if open.is_none() {
            let o = self.gpu.open_cb()?;
            for b in self.indirect.borrow().iter() {
                o.enc.useResource_usage(
                    ProtocolObject::from_ref(&**b),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
            }
            *open = Some(o);
        }
        Ok(())
    }

    /// Encode one dispatch of kernel `name` over `grid` threadgroups of `tg` threads with the
    /// given buffer bindings (index, buffer, byte offset) and an inline parameter struct.
    pub fn dispatch<P: Copy>(
        &self,
        name: &'static str,
        bufs: &[(usize, &Buf, usize)],
        params_index: usize,
        params: &P,
        grid: (usize, usize, usize),
        tg: (usize, usize, usize),
    ) -> Result<()> {
        let pso = self.gpu.pipeline(name);
        self.ensure_open()?;
        let mut open = self.open.borrow_mut();
        let enc = &open.as_ref().unwrap().enc;
        enc.setComputePipelineState(pso);
        for &(i, b, off) in bufs {
            debug_assert!(off < b.len, "buffer offset past end");
            // SAFETY: the buffer outlives the command buffer (the backend owns both until
            // `finish` returns); offset is within the buffer.
            unsafe { enc.setBuffer_offset_atIndex(Some(&b.raw), off, i) };
        }
        let nn = NonNull::from(params).cast::<c_void>();
        // SAFETY: `params` is a live `#[repr(C)]` value of `size_of::<P>()` bytes; Metal copies it.
        unsafe { enc.setBytes_length_atIndex(nn, std::mem::size_of::<P>(), params_index) };
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: grid.0,
                height: grid.1,
                depth: grid.2,
            },
            MTLSize {
                width: tg.0,
                height: tg.1,
                depth: tg.2,
            },
        );
        if let Some(prof) = &self.profile {
            let secs = open.take().unwrap().finish()?;
            prof.borrow_mut().push((name, secs));
        }
        Ok(())
    }

    /// Order every kernel encoded so far before every kernel encoded after (buffer scope).
    pub fn barrier(&self) {
        if self.profile.is_some() {
            return; // each dispatch already ran to completion
        }
        if let Some(o) = self.open.borrow().as_ref() {
            o.enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
    }

    /// Commit and wait; returns the GPU execution time of the command buffer in seconds (the sum
    /// of the per-dispatch times when profiling). Errors if the GPU reported a failure.
    pub fn finish(self) -> Result<f64> {
        let open = self.open.borrow_mut().take();
        let mut secs = match open {
            Some(o) => o.finish()?,
            None => 0.0,
        };
        if let Some(prof) = &self.profile {
            secs += prof.borrow().iter().map(|(_, s)| s).sum::<f64>();
        }
        Ok(secs)
    }

    /// The per-dispatch profile collected so far (empty unless `begin_profiled`).
    pub fn profile(&self) -> Vec<(&'static str, f64)> {
        self.profile
            .as_ref()
            .map(|p| p.borrow().clone())
            .unwrap_or_default()
    }
}

/// Threadgroups needed to cover `n` items with `per` per threadgroup.
pub fn groups(n: usize, per: usize) -> usize {
    n.div_ceil(per).max(1)
}
