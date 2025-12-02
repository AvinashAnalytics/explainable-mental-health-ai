"""
Comprehensive Test: Check ALL src/ files for:
1. Import errors
2. Syntax errors
3. Function availability
4. Relevance to project
"""

import os
import sys
import importlib.util
from pathlib import Path

# Simple output without special characters for Windows
def print_result(status, message):
    """Print test result with simple ASCII characters."""
    prefix = {
        'pass': '[PASS]',
        'fail': '[FAIL]',
        'warn': '[WARN]',
        'info': '[INFO]'
    }
    print(f"{prefix.get(status, '[INFO]')} {message}")

# Test results
test_results = {
    'passed': [],
    'failed': [],
    'to_delete': [],
    'to_fix': []
}

def test_file(file_path):
    """Test if a Python file can be imported and is relevant."""
    rel_path = os.path.relpath(file_path)
    print(f"\n{'='*80}")
    print(f"Testing: {rel_path}")
    print('='*80)
    
    # Read file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print_result('fail', f"Cannot read file: {e}")
        test_results['failed'].append((rel_path, f"Read error: {e}"))
        return False
    
    # Check if file is empty or just comments
    lines = [l.strip() for l in content.split('\n') if l.strip() and not l.strip().startswith('#')]
    if len(lines) == 0:
        print_result('warn', "File is empty or only has comments")
        test_results['to_delete'].append((rel_path, "Empty file"))
        return False
    
    # Check for syntax errors
    try:
        compile(content, file_path, 'exec')
        print_result('pass', "Syntax valid")
    except SyntaxError as e:
        print_result('fail', f"Syntax error: {e}")
        test_results['failed'].append((rel_path, f"Syntax error: {e}"))
        return False
    
    # Try to import the module
    try:
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules['test_module'] = module
            spec.loader.exec_module(module)
            
            # Check what's defined in the module
            exports = [name for name in dir(module) if not name.startswith('_')]
            if exports:
                print_result('pass', f"Import successful. Exports: {len(exports)} items")
                print_result('info', f"Items: {', '.join(exports[:10])}")
                test_results['passed'].append((rel_path, f"{len(exports)} exports"))
                return True
            else:
                print_result('warn', "File imports but exports nothing")
                test_results['to_delete'].append((rel_path, "No exports"))
                return False
    except Exception as e:
        error_msg = str(e)
        # Check if it's a missing dependency
        if "No module named" in error_msg:
            print_result('warn', f"Missing dependency: {error_msg}")
            test_results['to_fix'].append((rel_path, f"Missing dependency: {error_msg}"))
        else:
            print_result('fail', f"Import error: {error_msg}")
            test_results['failed'].append((rel_path, f"Import error: {error_msg}"))
        return False

def analyze_relevance(file_path):
    """Determine if file is relevant to the project."""
    relevant_keywords = [
        'depression', 'mental', 'health', 'classify', 'predict', 'explai',
        'dsm', 'phq', 'attention', 'lime', 'shap', 'llm', 'config',
        'train', 'model', 'data', 'load', 'preprocess', 'metric',
        'evaluation', 'calibration', 'safety', 'ethical'
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().lower()
        
        # Check for relevant keywords
        matches = [kw for kw in relevant_keywords if kw in content]
        return len(matches) > 0, matches
    except:
        return False, []

# Main test execution
print(f"\n{'='*80}")
print("COMPREHENSIVE SRC/ FOLDER ANALYSIS")
print(f"{'='*80}\n")

# Find all Python files in src/
src_path = Path("src")
py_files = list(src_path.rglob("*.py"))

print(f"Found {len(py_files)} Python files in src/\n")

# Test each file
for py_file in sorted(py_files):
    # Skip __pycache__ files
    if "__pycache__" in str(py_file):
        continue
    
    # Skip __init__.py for now (test separately)
    if py_file.name == "__init__.py":
        print(f"\n[INFO] Skipping __init__.py: {py_file}")
        continue
    
    test_file(str(py_file))
    
    # Check relevance
    is_relevant, keywords = analyze_relevance(str(py_file))
    if not is_relevant:
        print_result('warn', "File may not be relevant (no project keywords found)")

# Summary Report
print(f"\n\n{'='*80}")
print("SUMMARY REPORT")
print(f"{'='*80}\n")

print(f"[PASS] {len(test_results['passed'])} files")
for file, info in test_results['passed']:
    print(f"   {file} - {info}")

print(f"\n[FAIL] {len(test_results['failed'])} files")
for file, error in test_results['failed']:
    print(f"   {file} - {error}")

print(f"\n[FIX] {len(test_results['to_fix'])} files")
for file, issue in test_results['to_fix']:
    print(f"   {file} - {issue}")

print(f"\n[DELETE] {len(test_results['to_delete'])} files")
for file, reason in test_results['to_delete']:
    print(f"   {file} - {reason}")

# Overall status
total_issues = len(test_results['failed']) + len(test_results['to_fix']) + len(test_results['to_delete'])
total_files = len(test_results['passed']) + total_issues

print(f"\n{'='*80}")
print(f"OVERALL: {len(test_results['passed'])}/{total_files} files are working properly")
print(f"{'='*80}\n")

if total_issues == 0:
    print("SUCCESS: All src/ files are clean and working!\n")
else:
    print(f"WARNING: {total_issues} files need attention\n")

# Generate action items
print(f"\n{'='*80}")
print("ACTION ITEMS")
print(f"{'='*80}\n")

if test_results['to_delete']:
    print("Files to delete:")
    for file, reason in test_results['to_delete']:
        print(f"   Remove-Item '{file}' -Force  # {reason}")

if test_results['to_fix']:
    print("\nFiles to fix:")
    for file, issue in test_results['to_fix']:
        print(f"   # Fix: {file} - {issue}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
