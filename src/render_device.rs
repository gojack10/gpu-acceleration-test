use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum RenderDevice {
    CPU,
    GPU(String),
}

impl fmt::Display for RenderDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenderDevice::CPU => write!(f, "CPU"),
            RenderDevice::GPU(name) => write!(f, "GPU: {}", name),
        }
    }
} 