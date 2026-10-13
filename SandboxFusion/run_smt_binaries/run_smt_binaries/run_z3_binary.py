import subprocess
import tempfile
import os

def run_z3_on_smt2(smt2_content: str) -> str:
    # Create a temporary file with .smt2 extension
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.smt2', delete=False) as tmp_file:
        tmp_file.write(smt2_content)
        tmp_file_path = tmp_file.name

    try:
        # Run Z3 on the temporary file
        result = subprocess.run(['z3', tmp_file_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=False)

        # Combine stdout and stderr if needed
        output = result.stdout.strip()
        if result.stderr.strip():
            output += "\n[stderr]\n" + result.stderr.strip()
        return output

    finally:
        # Clean up the temporary file
        os.remove(tmp_file_path)

# Example usage:
if __name__ == "__main__":
    smt2_code = """
    (declare-const x Int)
    (assert (> x 5))
    (assert (< x 10))
    (check-sat)
    (get-model)
    """
    output = run_z3_on_smt2(smt2_code)
    print("Z3 Output:\n", output)
