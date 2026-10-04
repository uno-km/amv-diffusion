/**
 * @fileoverview Production Node.js client and workflow runner for termux-diffusion.
 * Supports both Method 1 (Direct CLI/Option generation) and Method 2 (ComfyUI DAG Workflow Runner).
 * Dual-format aware: seamlessly runs ComfyUI UI Graph Format (.json) & API Prompt Format (.json).
 *
 * @author uno-km (AMEVA Foundation)
 * @license Apache-2.0
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

/**
 * Executes termux-diffusion CLI with real-time NDJSON event streaming.
 * @private
 */
function _execCli(args, onProgress) {
  return new Promise((resolve, reject) => {
    const bin = 'termux-diffusion';
    const proc = spawn(bin, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdoutBuffer = '';
    let stderrBuffer = '';

    proc.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      stdoutBuffer += text;

      if (onProgress) {
        const lines = text.split('\n');
        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
            try {
              const event = JSON.parse(trimmed);
              onProgress(event);
            } catch (_) {
              // ignore partial json
            }
          }
        }
      }
    });

    proc.stderr.on('data', (chunk) => {
      stderrBuffer += chunk.toString();
    });

    proc.on('close', (code) => {
      if (code === 0) {
        resolve({
          success: true,
          stdout: stdoutBuffer.trim(),
          stderr: stderrBuffer.trim()
        });
      } else {
        const errMsg = stderrBuffer.trim() || stdoutBuffer.trim() || `Process exited with code ${code}`;
        reject(new Error(`[TermuxDiffusionError] ${errMsg}`));
      }
    });

    proc.on('error', (err) => {
      reject(new Error(`[TermuxDiffusionSpawnError] Failed to spawn '${bin}': ${err.message}`));
    });
  });
}

/**
 * [Method 1] Directly generate an AI image with plain options.
 *
 * @param {Object} options Generation options
 * @param {string} options.prompt Text description of image
 * @param {string} [options.model='realistic'] Model preset or .gguf path
 * @param {string} [options.device='auto'] Compute device ('auto', 'gpu', 'vulkan', 'cpu')
 * @param {number} [options.steps] Sampling steps
 * @param {number} [options.cfg] Guidance scale
 * @param {number} [options.width=512] Canvas width
 * @param {number} [options.height=512] Canvas height
 * @param {number} [options.seed=-1] RNG seed
 * @param {string} [options.scheduler] Noise scheduler ('flux', 'karras', 'ays', etc.)
 * @param {boolean} [options.detailFaces=false] Enable Mobile FaceDetailer
 * @param {string} [options.output] Destination image file path
 * @param {Function} [onProgress] Callback receiving structured NDJSON progress updates
 * @returns {Promise<Object>} Execution result with image path
 */
async function generate(options = {}, onProgress = null) {
  if (!options.prompt) {
    throw new Error("[InvalidParameter] 'prompt' is required for generate().");
  }

  const args = ['generate', options.prompt];

  if (options.model) args.push('-m', options.model);
  if (options.device) args.push('-d', options.device);
  if (options.steps !== undefined) args.push('-s', String(options.steps));
  if (options.cfg !== undefined) args.push('-c', String(options.cfg));
  if (options.width) args.push('-W', String(options.width));
  if (options.height) args.push('-H', String(options.height));
  if (options.seed !== undefined) args.push('--seed', String(options.seed));
  if (options.scheduler) args.push('--scheduler', options.scheduler);
  if (options.output) args.push('-o', options.output);
  if (options.detailFaces) args.push('--detail-faces');
  if (options.safeMode || options.safe_mode || options.keepAppsAlive || options.keep_apps_alive) args.push('--safe-mode');
  if (options.gpuDutyCycle || options.gpu_duty_cycle) args.push('--gpu-duty-cycle', String(options.gpuDutyCycle || options.gpu_duty_cycle));

  if (onProgress) {
    args.push('--json-progress');
  }

  const res = await _execCli(args, onProgress);
  return res;
}

/**
 * [Method 2] Run a ComfyUI Workflow JSON file (UI or API format) directly by path.
 *
 * @param {string} workflowPath Path to ComfyUI workflow .json file
 * @param {Object} [options={}] Execution options
 * @param {string} [options.outputDir] Output directory for generated images
 * @param {string} [options.device='auto'] Compute device ('auto', 'gpu', 'vulkan', 'cpu')
 * @param {boolean} [options.forceRun=false] Bypass hardware checks for 16GB/24GB+ devices
 * @param {boolean} [options.safeMode=false] Preserve SurfaceFlinger UI and background apps
 * @param {number} [options.gpuDutyCycle=90] Target GPU duty cycle percentage
 * @param {Function} [onProgress] Progress event listener
 * @returns {Promise<Object>} Execution result
 */
async function runWorkflow(workflowPath, options = {}, onProgress = null) {
  if (!workflowPath) {
    throw new Error("[InvalidParameter] 'workflowPath' is required.");
  }
  const resolvedPath = path.resolve(workflowPath);
  if (!fs.existsSync(resolvedPath)) {
    throw new Error(`[FileNotFound] Workflow file not found: ${resolvedPath}`);
  }

  const args = ['workflow', 'run', resolvedPath];
  if (options.outputDir) args.push('-o', options.outputDir);
  if (options.device) args.push('-d', options.device);
  if (options.forceRun) args.push('--force-run');
  if (options.safeMode || options.safe_mode || options.keepAppsAlive || options.keep_apps_alive) args.push('--safe-mode');
  if (options.gpuDutyCycle || options.gpu_duty_cycle) args.push('--gpu-duty-cycle', String(options.gpuDutyCycle || options.gpu_duty_cycle));
  if (onProgress) args.push('--json-progress');

  const res = await _execCli(args, onProgress);
  return res;
}

/**
 * Inspects a ComfyUI workflow JSON file to inspect DAG structure and execution order.
 *
 * @param {string} workflowPath Path to ComfyUI workflow .json file
 * @returns {Promise<Object>} Inspection summary
 */
async function inspectWorkflow(workflowPath) {
  const resolvedPath = path.resolve(workflowPath);
  if (!fs.existsSync(resolvedPath)) {
    throw new Error(`[FileNotFound] Workflow file not found: ${resolvedPath}`);
  }
  const args = ['workflow', 'inspect', resolvedPath];
  const res = await _execCli(args);
  return res;
}

/**
 * Launch the AmfyUI Mobile Studio & Web Daemon.
 *
 * @param {Object} [options={}] Server launch options
 * @param {number} [options.port=11553] Server listening port
 * @param {string} [options.host='0.0.0.0'] Host interface to bind
 * @returns {ChildProcess} Spawned process instance
 */
function launchAmfyUI(options = {}) {
  const port = options.port || 11553;
  const host = options.host || '0.0.0.0';
  const args = ['web', '-p', String(port), '-H', host];
  return spawn('termux-diffusion', args, { stdio: 'inherit' });
}

module.exports = {
  generate,
  runWorkflow,
  inspectWorkflow,
  launchAmfyUI
};
