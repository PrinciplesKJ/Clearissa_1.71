#!/bin/bash
# ========================================
# Clearissa macOS Application Builder
# ========================================
# This script builds a standalone macOS application bundle for Clearissa.
# It creates a .app bundle that can be distributed and run on macOS systems.
#
# PREREQUISITES:
#   - macOS 10.13 (High Sierra) or later
#   - Python 3.9, 3.10, or 3.11 installed
#   - Xcode Command Line Tools (install with: xcode-select --install)
#
# USAGE:
#   ./build_clearissa.sh              - Build with automatic virtual environment
#   ./build_clearissa.sh --no-venv    - Use current Python environment
#   ./build_clearissa.sh --clean      - Clean previous builds first
#   ./build_clearissa.sh --help       - Show this help message
#
# FIRST TIME SETUP:
#   1. Open Terminal and navigate to the Clearissa project directory
#   2. Make this script executable:
#      chmod +x build_clearissa.sh
#   3. Run the script:
#      ./build_clearissa.sh
#
# OUTPUT:
#   The build process creates:
#   - dist/Clearissa.app  - The application bundle (drag to Applications folder)
#   - build/              - Temporary build files (can be deleted)
#
# DISTRIBUTION:
#   To share the application:
#   1. Compress the .app bundle:
#      Right-click on dist/Clearissa.app → Compress "Clearissa.app"
#      OR use terminal: cd dist && zip -r Clearissa.zip Clearissa.app
#   2. Share the resulting Clearissa.zip file
#
#   Note: Users may need to allow the app in System Preferences → Security & Privacy
#   on first launch if it's not code-signed.
#
# TROUBLESHOOTING:
#   - If you get "Permission denied": Run chmod +x build_clearissa.sh
#   - If Python is not found: Install Python from https://www.python.org/downloads/
#   - If build fails: Try running with --clean flag to remove old build files
#   - For dependency issues: Delete .venv folder and run script again
#
# ========================================

set -u  # Exit on undefined variable

# Colours for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Colour

# Helper functions for coloured output
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_info() {
    echo -e "${GREEN}[*]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

# Show help message
show_help() {
    grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Parse command line arguments
USE_VENV=1
DO_CLEAN=0

for arg in "$@"; do
    case $arg in
        --no-venv)
            USE_VENV=0
            shift
            ;;
        --clean)
            DO_CLEAN=1
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

print_header "Clearissa macOS Application Builder"

# Change to the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Store important paths
ROOT_DIR="$PWD"
VENV_DIR="$ROOT_DIR/.venv"
SPEC_FILE="$ROOT_DIR/Clearissa.spec"
DIST_DIR="$ROOT_DIR/dist"
BUILD_DIR="$ROOT_DIR/build"

print_info "Working directory: $ROOT_DIR"

# Check if spec file exists
if [ ! -f "$SPEC_FILE" ]; then
    print_error "Clearissa.spec not found at: $SPEC_FILE"
    echo "Please ensure you are running this script from the Clearissa project root."
    exit 1
fi

print_success "Found Clearissa.spec"

# Find Python interpreter
print_info "Searching for Python installation..."

PYTHON_CMD=""

# Try python3.11 first (preferred)
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    # Check if it's Python 3 (use grep -E for macOS compatibility)
    if "python" --version 2>&1 | grep -qE 'Python 3\.[0-9]+'; then
        PYTHON_CMD="python"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    print_error "Python 3.9 or later not found."
    echo "Please install Python from: https://www.python.org/downloads/"
    echo "Or use Homebrew: brew install python@3.11"
    exit 1
fi

print_success "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo

# Setup virtual environment if requested
if [ "$USE_VENV" -eq 1 ]; then
    print_info "Setting up virtual environment..."

    if [ ! -f "$VENV_DIR/bin/python" ]; then
        echo "    Creating new virtual environment at: .venv"
        if ! $PYTHON_CMD -m venv "$VENV_DIR"; then
            print_error "Failed to create virtual environment"
            exit 1
        fi
        print_success "Virtual environment created"
    else
        echo "    Using existing virtual environment at: .venv"
    fi

    # Activate the virtual environment
    if ! source "$VENV_DIR/bin/activate"; then
        print_error "Failed to activate virtual environment"
        exit 1
    fi

    PYTHON_CMD="python"
    print_success "Virtual environment activated"
    echo
else
    print_info "Using current Python environment (--no-venv specified)"
    echo
fi

# Upgrade pip
print_info "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip --quiet || print_warning "Failed to upgrade pip, continuing anyway..."
print_success "pip upgraded"
echo

# Install requirements
print_info "Installing dependencies from requirements.txt..."
if [ -f "$ROOT_DIR/requirements.txt" ]; then
    if ! $PYTHON_CMD -m pip install -r "$ROOT_DIR/requirements.txt" --quiet; then
        print_error "Failed to install requirements"
        exit 1
    fi
    print_success "Dependencies installed successfully"
else
    print_warning "requirements.txt not found"
fi
echo

# Verify optional plotting libraries
print_info "Verifying optional plotting libraries (plotnine, mizani)..."
$PYTHON_CMD -c "import plotnine; import mizani" &> /dev/null
if [ $? -ne 0 ]; then
    print_warning "Optional plotting packages (plotnine, mizani) are not available."
    echo "          ggplot export will be disabled in the application unless you install them."
    echo "          Install with: pip install plotnine mizani"
    echo
else
    print_success "Optional plotting libraries are available"
    echo
fi

# Install PyInstaller
print_info "Installing PyInstaller..."
if ! $PYTHON_CMD -m pip install "pyinstaller>=6.0" --quiet; then
    print_error "Failed to install PyInstaller"
    exit 1
fi
print_success "PyInstaller installed successfully"
echo

# Clean Python cache files
print_info "Cleaning Python cache files..."
find "$ROOT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$ROOT_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$ROOT_DIR" -type f -name "*.pyo" -delete 2>/dev/null || true
print_success "Python cache cleaned"
echo

# Clean previous builds
print_info "Cleaning previous builds..."
if [ -d "$BUILD_DIR" ]; then
    echo "    Removing: build/"
    rm -rf "$BUILD_DIR"
fi
if [ -d "$DIST_DIR" ]; then
    echo "    Removing: dist/"
    rm -rf "$DIST_DIR"
fi
print_success "Clean completed"
echo

# Set environment variable for PyQtGraph
export PYQTGRAPH_QT_LIB=PyQt5

# Run PyInstaller
print_info "Building macOS application with PyInstaller..."
echo "    This may take several minutes..."
echo

if ! $PYTHON_CMD -m PyInstaller --clean --noconfirm "$SPEC_FILE"; then
    echo
    print_header "BUILD FAILED"
    print_error "PyInstaller encountered an error."
    echo "Check the output above for details."
    exit 1
fi

# Check if application bundle was created
APP_PATH="$DIST_DIR/Clearissa.app"
if [ ! -d "$APP_PATH" ]; then
    echo
    print_header "BUILD FAILED"
    print_error "Application bundle not found at expected location:"
    echo "$APP_PATH"
    exit 1
fi

# Success!
echo
print_header "BUILD SUCCESSFUL!"

echo -e "${GREEN}Application created at:${NC}"
echo "$APP_PATH"
echo

# Get application size
APP_SIZE=$(du -sh "$APP_PATH" | cut -f1)
echo -e "${GREEN}Application size:${NC} $APP_SIZE"
echo

echo -e "${BLUE}NEXT STEPS:${NC}"
echo
echo "1. Test the application:"
echo "   open \"$APP_PATH\""
echo
echo "2. Move to Applications folder (optional):"
echo "   cp -r \"$APP_PATH\" /Applications/"
echo
echo "3. Create distributable package:"
echo "   cd dist"
echo "   zip -r Clearissa.zip Clearissa.app"
echo
echo -e "${YELLOW}NOTE:${NC} On first launch, macOS may show a security warning."
echo "      Go to System Preferences → Security & Privacy → General"
echo "      and click 'Open Anyway' to allow the application to run."
echo

# Deactivate venv if we used one
if [ "$USE_VENV" -eq 1 ]; then
    deactivate 2>/dev/null || true
fi

print_success "Build process complete!"
echo

exit 0