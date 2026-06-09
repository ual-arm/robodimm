import { RobotSpec, ProgramSpec, TorqueLog } from '../model/schemas';

let activeBackendUrl = import.meta.env.VITE_PRO_BACKEND_URL_PRIMARY || 'http://127.0.0.1:8001';

export function setApiBackendUrl(url: string) {
  activeBackendUrl = url;
}

export function getApiBackendUrl(): string {
  return activeBackendUrl;
}

// Helper for fetch with AbortController timeout
export async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs = 800
): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(id);
    return response;
  } catch (err) {
    clearTimeout(id);
    throw err;
  }
}

export interface BackendCheckResult {
  connected: boolean;
  version: string | null;
  pinocchioVersion: string | null;
  licenseStatus: string | null;
  capabilities: any | null;
  incompatible: boolean;
  url: string;
}

export async function pingBackendUrl(url: string, timeoutMs = 800): Promise<BackendCheckResult> {
  try {
    const healthResp = await fetchWithTimeout(`${url}/api/health`, {}, timeoutMs);
    if (!healthResp.ok) {
      return { connected: false, version: null, pinocchioVersion: null, licenseStatus: null, capabilities: null, incompatible: false, url };
    }
    const healthData = await healthResp.json();
    const version = healthData.version || null;
    const pinocchioVersion = healthData.pinocchio_version || null;

    const capResp = await fetchWithTimeout(`${url}/api/capabilities`, {}, timeoutMs);
    if (!capResp.ok) {
      return { connected: true, version, pinocchioVersion, licenseStatus: null, capabilities: null, incompatible: false, url };
    }
    const capData = await capResp.json();
    const capabilities = capData.capabilities || null;
    const licenseStatus = capData.license_status || null;

    // Check compatibility: verify CR4 and CR6 capabilities are present
    const incompatible = !capabilities || !capabilities.CR4 || !capabilities.CR6;

    return {
      connected: true,
      version,
      pinocchioVersion,
      licenseStatus,
      capabilities,
      incompatible,
      url
    };
  } catch {
    return { connected: false, version: null, pinocchioVersion: null, licenseStatus: null, capabilities: null, incompatible: false, url };
  }
}

export async function calculateBackendDynamics(
  robot: RobotSpec,
  program: ProgramSpec
): Promise<TorqueLog | null> {
  try {
    const resp = await fetchWithTimeout(`${activeBackendUrl}/api/dynamics`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ robot, program })
    }, 15000); // 15 seconds timeout for legacy program calculations
    if (resp.ok) {
      return await resp.json();
    }
  } catch (err) {
    console.error('FastAPI Python backend legacy calculation failed:', err);
  }
  return null;
}

export async function calculateBackendBatchDynamics(
  robot: RobotSpec,
  samples: Array<{ time_s: number; q: number[]; qd: number[]; qdd: number[] }>
): Promise<any | null> {
  try {
    const resp = await fetchWithTimeout(`${activeBackendUrl}/api/dynamics/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        robot,
        samples,
        schema_version: 'robodimm.dynamics.v1'
      })
    }, 120000); // 120 seconds timeout: Pinocchio model cold start can take 30+ seconds
    if (resp.ok) {
      return await resp.json();
    } else {
      const errText = await resp.text();
      console.error('FastAPI batch dynamics calculation response error:', errText);
    }
  } catch (err) {
    console.error('FastAPI batch dynamics calculation connection failed:', err);
  }
  return null;
}

export async function uploadPackageToBackend(
  files: File[]
): Promise<RobotSpec | null> {
  try {
    const formData = new FormData();
    for (const file of files) {
      const path = file.webkitRelativePath || file.name;
      formData.append('files', file, path);
    }

    const resp = await fetch(`${activeBackendUrl}/api/packages/upload`, {
      method: 'POST',
      body: formData
    });

    if (resp.ok) {
      return await resp.json();
    } else {
      const errText = await resp.text();
      console.error('Failed to upload package to backend:', errText);
    }
  } catch (err) {
    console.error('Error uploading package:', err);
  }
  return null;
}
