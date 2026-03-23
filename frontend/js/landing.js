/**
 * landing.js - Landing Page Logic
 * ================================
 * 
 * Handles mode selection, URL generation, and interactive elements.
 * 
 * IMPORTANT: PRO mode requires the backend to be running on port 8000.
 * DEMO mode works on any port (frontend-only).
 */

// Determine the correct URLs based on current host
function getSimulatorUrls() {
  const currentPort = window.location.port;
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;
  
  // DEMO mode: works on current port (relative URL)
  const demoUrl = 'simulator.html?mode=demo';
  
  // PRO mode: always needs port 8000 (backend)
  // If we're already on 8000, use relative URL
  // If we're on another port (e.g., 8080), explicitly go to 8000
  let proUrl;
  if (currentPort === '8000' || currentPort === '') {
    // Already on backend port (or default port)
    proUrl = 'simulator?mode=pro';
  } else {
    // Need to switch to backend port
    proUrl = `${protocol}//${hostname}:8000/simulator?mode=pro`;
  }
  
  return { demoUrl, proUrl, currentPort };
}

// Update links on page load
document.addEventListener('DOMContentLoaded', () => {
  const { demoUrl, proUrl, currentPort } = getSimulatorUrls();
  
  // Update DEMO link (always relative)
  const demoLink = document.getElementById('demo-link');
  if (demoLink) {
    demoLink.href = demoUrl;
  }
  
  // Update PRO link (may be absolute if on wrong port)
  const proLink = document.getElementById('pro-link');
  if (proLink) {
    proLink.href = proUrl;
    
    // Add click handler
    proLink.addEventListener('click', (e) => {
      trackModeSelection('pro');
    });
  }
  
  // Add keyboard navigation
  document.addEventListener('keydown', (e) => {
    if (e.key === '1') {
      window.location.href = demoUrl;
    } else if (e.key === '2') {
      window.location.href = proUrl;
    }
  });
});

// Check if the user has a preferred mode saved
function getPreferredMode() {
  try {
    return localStorage.getItem('robodimm_preferred_mode');
  } catch (e) {
    return null;
  }
}

// Save preferred mode
function savePreferredMode(mode) {
  try {
    localStorage.setItem('robodimm_preferred_mode', mode);
  } catch (e) {
    // Silently fail
  }
}

// Track mode selection
function trackModeSelection(mode) {
  savePreferredMode(mode);
}

// No export needed for browser usage
