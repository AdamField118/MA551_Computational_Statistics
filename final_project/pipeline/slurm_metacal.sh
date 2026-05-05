#!/usr/bin/env bash
# =============================================================================
# slurm_metacal.sh
# Submit from MA551_Computational_Statistics/ with:
#   sbatch final_project/pipeline/slurm_metacal.sh
# =============================================================================

#SBATCH --job-name=metacal_bias
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=96G
#SBATCH --gres=gpu:0
#SBATCH --time=01:00:00
#SBATCH --output=final_project/logs/metacal_%j.out
#SBATCH --error=final_project/logs/metacal_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adfield@wpi.edu

# ---------------------------------------------------------------------------
# Paths  -- all relative to MA551_Computational_Statistics/ (the project root)
# SLURM sets the working directory to wherever you called sbatch from,
# so as long as you run  sbatch final_project/pipeline/slurm_metacal.sh
# from the project root these paths resolve correctly.
# ---------------------------------------------------------------------------
PROJ_ROOT="$(pwd)"   # MA551_Computational_Statistics/
CATALOG="${PROJ_ROOT}/cosmos15_superbit2023_phot_shapes_with_sigma.csv"
PSFEX_DIR="${PROJ_ROOT}/psf_data/emp_psfs_best/psfex-output"

# Pick the first .psf file found in psf_data/
# Change this to a specific file if you want a particular exposure:
#   PSFEX="${PSFEX_DIR}/Abell3411_1_300_1684688714_clean_starcat.psf"
PSFEX="$(find "${PSFEX_DIR}" -maxdepth 1 -name '*.psf' | head -1)"

OUTDIR="${PROJ_ROOT}/final_project/results"
PIPELINE="${PROJ_ROOT}/final_project/pipeline/metacal_pipeline.py"

mkdir -p "${OUTDIR}"
mkdir -p "${PROJ_ROOT}/final_project/logs"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Turing conda setup (same path used in all ShearNet SLURM scripts)
source /cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/miniconda3-25.1.1-24g7bpuxyyxo5pfd4zn5sldbomvz736a/etc/profile.d/conda.sh
conda activate shearnet_gpu

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "=============================================="
echo "Job  : ${SLURM_JOB_ID}"
echo "Node : ${SLURMD_NODENAME}"
echo "CPUs : ${SLURM_CPUS_PER_TASK}"
echo "Root : ${PROJ_ROOT}"
echo "Cat  : ${CATALOG}"
echo "PSF  : ${PSFEX:-none (Gaussian fallback)}"
echo "Start: $(date)"
echo "=============================================="

# ---------------------------------------------------------------------------
# Python step: metacalibration
# ---------------------------------------------------------------------------
start=$(date +%s)

if [ -n "${PSFEX}" ] && [ -f "${PSFEX}" ]; then
    python "${PIPELINE}" \
        --catalog   "${CATALOG}"  \
        --n-obs     50000         \
        --n-workers "${SLURM_CPUS_PER_TASK}" \
        --psfex     "${PSFEX}"    \
        --n-jack    20            \
        --seed      150           \
        --outdir    "${OUTDIR}"
else
    echo "WARNING: No .psf file found in ${PSFEX_DIR}. Using Gaussian PSF."
    python "${PIPELINE}" \
        --catalog   "${CATALOG}"  \
        --n-obs     50000         \
        --n-workers "${SLURM_CPUS_PER_TASK}" \
        --psf-fwhm  0.5           \
        --n-jack    20            \
        --seed      150           \
        --outdir    "${OUTDIR}"
fi

PY_EXIT=$?
echo "Python exit code: ${PY_EXIT}"

# ---------------------------------------------------------------------------
# R step: full statistical analysis (reads metacal_bias.json automatically)
# ---------------------------------------------------------------------------
if [ ${PY_EXIT} -eq 0 ]; then
    echo ""
    echo "=== Running R analysis pipeline ==="
    module load R/4.3.1 2>/dev/null || true   # load if available; harmless if not
    Rscript "${PROJ_ROOT}/final_project/R/run_all.R"
else
    echo "Python step failed -- skipping R analysis."
fi

end=$(date +%s)
runtime=$(( end - start ))
printf -v h "%02d" $(( runtime / 3600 ))
printf -v m "%02d" $(( (runtime % 3600) / 60 ))
printf -v s "%02d" $(( runtime % 60 ))
echo ""
echo "End    : $(date)"
echo "Runtime: ${h}:${m}:${s}"
