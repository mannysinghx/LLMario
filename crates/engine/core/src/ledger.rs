//! The engine's own memory ledger (Architecture §5.1).
//!
//! OS process counters mislead (shared file-backed pages count as RSS on Linux and Windows; macOS
//! lists mapped weights under "Cached Files"; Metal wired memory is invisible to `ps`), so the engine
//! keeps its own per-device table and exposes it through `GET /engine/ledger`. Every allocation and
//! release in the engine updates it; nothing is counted twice.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum DeviceId {
    Host,
    Gpu(u8),
    Npu(u8),
}

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceId::Host => write!(f, "host"),
            DeviceId::Gpu(i) => write!(f, "gpu{i}"),
            DeviceId::Npu(i) => write!(f, "npu{i}"),
        }
    }
}

/// One device's rows. All values are bytes.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceLedger {
    pub device: Option<DeviceId>,
    /// Model file bytes mapped (file-backed, evictable by the OS).
    pub weights_mapped: u64,
    /// Sampled estimate of mapped bytes currently resident (labelled an estimate).
    pub weights_resident_est: u64,
    /// Pinned bytes: Metal residency sets, mlock, device-local copies.
    pub weights_wired: u64,
    /// Anonymous copies: repacked tiles, in-situ-quantised tensors.
    pub weights_private: u64,
    pub kv_arena_reserved: u64,
    pub kv_arena_in_use: u64,
    pub recurrent_state: u64,
    pub scratch_reserved: u64,
    pub prompt_cache: u64,
    pub repack_cache_mapped: u64,
    pub grammar_cache: u64,
    pub drafter: u64,
    pub runtime_fixed: u64,
    /// Ceiling the planner used for this device.
    pub budget: u64,
    pub headroom: u64,
}

impl DeviceLedger {
    /// Bytes that count against the budget: everything the engine itself holds, with mapped
    /// weights counted at their mapped size (the conservative choice; the OS may hold less).
    pub fn charged(&self) -> u64 {
        self.weights_mapped
            + self.weights_private
            + self.kv_arena_reserved
            + self.recurrent_state
            + self.scratch_reserved
            + self.prompt_cache
            + self.grammar_cache
            + self.drafter
            + self.runtime_fixed
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct LedgerSnapshot {
    pub devices: Vec<DeviceLedger>,
    pub planned_peak: u64,
    pub measured_peak: u64,
    pub pressure: PressureState,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PressureState {
    #[default]
    Normal,
    Warning,
    Critical,
}

/// Thread-safe ledger shared by the loader, the KV arena, the caches and the server.
#[derive(Default)]
pub struct Ledger {
    devices: Mutex<Vec<DeviceLedger>>,
    planned_peak: AtomicU64,
    measured_peak: AtomicU64,
    pressure: Mutex<PressureState>,
}

impl Ledger {
    pub fn new(devices: Vec<DeviceId>) -> Ledger {
        let devices = devices
            .into_iter()
            .map(|d| DeviceLedger {
                device: Some(d),
                ..Default::default()
            })
            .collect();
        Ledger {
            devices: Mutex::new(devices),
            ..Default::default()
        }
    }

    /// Apply a change to one device's rows.
    pub fn update(&self, device: DeviceId, f: impl FnOnce(&mut DeviceLedger)) {
        let mut devs = self.devices.lock().unwrap();
        if let Some(d) = devs.iter_mut().find(|d| d.device == Some(device)) {
            f(d);
        } else {
            let mut d = DeviceLedger {
                device: Some(device),
                ..Default::default()
            };
            f(&mut d);
            devs.push(d);
        }
    }

    pub fn set_planned_peak(&self, bytes: u64) {
        self.planned_peak.store(bytes, Ordering::Relaxed);
    }

    pub fn record_measured_peak(&self, bytes: u64) {
        self.measured_peak.fetch_max(bytes, Ordering::Relaxed);
    }

    pub fn set_pressure(&self, p: PressureState) {
        *self.pressure.lock().unwrap() = p;
    }

    pub fn snapshot(&self) -> LedgerSnapshot {
        LedgerSnapshot {
            devices: self.devices.lock().unwrap().clone(),
            planned_peak: self.planned_peak.load(Ordering::Relaxed),
            measured_peak: self.measured_peak.load(Ordering::Relaxed),
            pressure: *self.pressure.lock().unwrap(),
        }
    }

    pub fn charged(&self, device: DeviceId) -> u64 {
        self.devices
            .lock()
            .unwrap()
            .iter()
            .find(|d| d.device == Some(device))
            .map(|d| d.charged())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_updates_and_snapshots() {
        let l = Ledger::new(vec![DeviceId::Host]);
        l.update(DeviceId::Host, |d| d.weights_mapped += 1000);
        l.update(DeviceId::Host, |d| d.kv_arena_reserved += 500);
        l.update(DeviceId::Gpu(0), |d| d.weights_wired += 7);
        assert_eq!(l.charged(DeviceId::Host), 1500);
        let s = l.snapshot();
        assert_eq!(s.devices.len(), 2);
        assert_eq!(s.devices[1].device, Some(DeviceId::Gpu(0)));
        l.record_measured_peak(10);
        l.record_measured_peak(5);
        assert_eq!(l.snapshot().measured_peak, 10);
    }
}
