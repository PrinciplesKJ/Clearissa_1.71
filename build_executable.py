#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clearissa Build Script
======================

Automated build script for creating standalone executables with PyInstaller.
Handles platform-specific configurations and validates build output.

Usage:
    python build_executable.py [--clean] [--onefile] [--debug]

Options:
    --clean     Remove build directories before building
    --onefile   Create single executable file (default: onedir)
    --debug     Enable console window and debug output

Author: Križan Jurinović
Date: November 2025
"""

import os
import sys
import shutil
import argparse
import subprocess
import platform
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except Exception:
        pass  # Fallback to default encoding


class ClearissaBuilder:
    """Automated build orchestrator for Clearissa executable."""

    def __init__(self, clean=False, onefile=False, debug=False):
        self.clean = clean
        self.onefile = onefile
        self.debug = debug
        self.project_root = Path(__file__).parent
        self.spec_file = self.project_root / "Clearissa.spec"

    def print_header(self, message):
        """Print formatted header."""
        print("\n" + "=" * 70)
        print(f"  {message}")
        print("=" * 70 + "\n")

    def check_requirements(self):
        """Verify all required files and dependencies exist."""
        self.print_header("Checking Build Requirements")

        # Check required files
        required_files = [
            "Clearissa.spec",
            "resource_utils.py",
            "manual.html",
            "clearissa_logo.png",
            "clearissa_icon.ico",
            "core/Clearissa_main.py",
        ]

        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"[OK] Found: {file_path}")
            else:
                print(f"[X] Missing: {file_path}")
                missing_files.append(file_path)

        if missing_files:
            print(f"\n[ERROR] Missing required files:")
            for file_path in missing_files:
                print(f"   - {file_path}")
            sys.exit(1)

        # Check PyInstaller
        try:
            import PyInstaller
            print(f"[OK] PyInstaller {PyInstaller.__version__} installed")
        except ImportError:
            print("[X] PyInstaller not found")
            print("\n[ERROR] PyInstaller is required. Install with:")
            print("   pip install pyinstaller")
            sys.exit(1)

        # Check critical dependencies
        critical_deps = [
            "PyQt5", "numpy", "pandas", "scipy", "matplotlib",
            "pyqtgraph", "openpyxl", "plotnine"
        ]

        missing_deps = []
        for dep in critical_deps:
            try:
                __import__(dep)
                print(f"[OK] {dep} installed")
            except ImportError:
                print(f"[X] {dep} not found")
                missing_deps.append(dep)

        if missing_deps:
            print(f"\n[ERROR] Missing required dependencies:")
            for dep in missing_deps:
                print(f"   - {dep}")
            print("\nInstall missing dependencies with:")
            print(f"   pip install {' '.join(missing_deps)}")
            sys.exit(1)

        print("\n[OK] All requirements satisfied\n")

    def clean_build_directories(self):
        """Remove build, dist, and __pycache__ directories."""
        self.print_header("Cleaning Build Directories")

        directories_to_clean = [
            self.project_root / "build",
            self.project_root / "dist",
        ]

        for dir_path in directories_to_clean:
            if dir_path.exists():
                print(f"Removing: {dir_path}")
                shutil.rmtree(dir_path)
            else:
                print(f"Already clean: {dir_path}")

        # Clean __pycache__ directories
        for pycache in self.project_root.rglob("__pycache__"):
            print(f"Removing: {pycache}")
            shutil.rmtree(pycache)

        print("\n[OK] Build directories cleaned\n")

    def modify_spec_for_onefile(self):
        """Modify spec file for onefile mode if requested."""
        if not self.onefile:
            return

        self.print_header("Configuring for Single-File Build")
        print("[WARNING] Note: Onefile mode creates a single executable but has slower startup time\n")

    def build_executable(self):
        """Run PyInstaller with the spec file."""
        self.print_header("Building Executable with PyInstaller")

        # Build PyInstaller command
        cmd = ["pyinstaller"]

        if self.clean:
            cmd.append("--clean")

        if self.debug:
            cmd.append("--debug=all")

        cmd.append(str(self.spec_file))

        print(f"Running: {' '.join(cmd)}\n")

        # Run PyInstaller
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            print(result.stdout)
            print("\n[OK] Build completed successfully\n")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Build failed with error:")
            print(e.stdout)
            sys.exit(1)

    def verify_build_output(self):
        """Verify the build created expected files."""
        self.print_header("Verifying Build Output")

        dist_dir = self.project_root / "dist" / "Clearissa"

        if not dist_dir.exists():
            print(f"[X] Distribution directory not found: {dist_dir}")
            sys.exit(1)

        # Check for executable
        executable_name = "Clearissa.exe" if platform.system() == "Windows" else "Clearissa"
        executable_path = dist_dir / executable_name

        if executable_path.exists():
            size_mb = executable_path.stat().st_size / (1024 * 1024)
            print(f"[OK] Executable created: {executable_path}")
            print(f"  Size: {size_mb:.1f} MB")
        else:
            print(f"[X] Executable not found: {executable_path}")
            sys.exit(1)

        # Check for bundled resources
        expected_resources = [
            "manual.html",
            "clearissa_logo.png",
            "clearissa_icon.ico",
            "resource_utils.py",
        ]

        for resource in expected_resources:
            resource_path = dist_dir / resource
            if resource_path.exists():
                print(f"[OK] Resource bundled: {resource}")
            else:
                print(f"[WARNING] Resource not found: {resource}")

        print("\n[OK] Build verification complete\n")

    def print_summary(self):
        """Print build summary and next steps."""
        self.print_header("Build Summary")

        dist_dir = self.project_root / "dist" / "Clearissa"
        executable_name = "Clearissa.exe" if platform.system() == "Windows" else "Clearissa"

        print(f"Platform: {platform.system()} {platform.machine()}")
        print(f"Build type: {'Single file' if self.onefile else 'Directory'}")
        print(f"Debug mode: {'Enabled' if self.debug else 'Disabled'}")
        print(f"\nOutput location: {dist_dir}")
        print(f"Executable: {executable_name}")

        print("\n" + "─" * 70)
        print("Next Steps:")
        print("─" * 70)
        print(f"1. Test the executable:")
        print(f"   cd {dist_dir}")
        print(f"   ./{executable_name}")
        print()
        print("2. Verify critical functionality:")
        print("   - Application launches without errors")
        print("   - Manual (Help) opens correctly")
        print("   - Config files are created in user directory:")
        if platform.system() == "Windows":
            print("     Windows: %APPDATA%\\Clearissa\\config\\")
        else:
            print("     Linux/macOS: ~/.clearissa/config/")
        print("   - CSV data import works correctly")
        print("   - Conversion operations function properly")
        print("   - Plots render correctly")
        print()
        print("3. Package for distribution:")
        print(f"   - Zip the entire '{dist_dir.name}' directory")
        print("   - Include README with system requirements")
        print()

    def run(self):
        """Execute the complete build process."""
        self.print_header(f"Clearissa Executable Build - {platform.system()}")

        try:
            self.check_requirements()

            if self.clean:
                self.clean_build_directories()

            self.modify_spec_for_onefile()
            self.build_executable()
            self.verify_build_output()
            self.print_summary()

            print("[OK] Build process completed successfully!\n")

        except KeyboardInterrupt:
            print("\n\n[ERROR] Build cancelled by user\n")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n[ERROR] Unexpected error during build: {e}\n")
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build Clearissa standalone executable with PyInstaller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_executable.py                    # Standard build
  python build_executable.py --clean            # Clean build
  python build_executable.py --onefile          # Single file executable
  python build_executable.py --debug            # Debug build with console
        """
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove build directories before building"
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Create single executable file (slower startup)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable console window and debug output"
    )

    args = parser.parse_args()

    builder = ClearissaBuilder(
        clean=args.clean,
        onefile=args.onefile,
        debug=args.debug
    )
    builder.run()


if __name__ == "__main__":
    main()
