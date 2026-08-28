import csv

from qrisp import QuantumSession


def analyze_circuit(qs: QuantumSession):
    res = qs.compile().transpile()

    data = res.count_ops()
    gate_count = sum(data.values())

    result = {
        "num_qubits": res.num_qubits(),
        "depth": res.depth(),
        "gate_count": gate_count,
        "cnot_total": res.cnot_count(),
        "x_count": data.get("x", 0),
        "h_count": data.get("h", 0),
        "u3_count": data.get("u3", 0),
        "cx_count": data.get("cx", 0),
        "t_count": data.get("t", 0),
        "t_dg_count": data.get("t_dg", 0),
        "p_count": data.get("p", 0),
    }

    print("NUMBER OF QUBITS: ", result["num_qubits"], "\n")
    print("DEPTH OF CIRCUIT: ", result["depth"], "\n")
    print("NUMBER OF GATES: ", result["gate_count"], "\n")
    print("NUMBER OF X-GATES: ", result["x_count"], "\n")
    print("NUMBER OF HADAMARD-GATES: ", result["h_count"], "\n")
    print("NUMBER OF U3-GATES: ", result["u3_count"], "\n")
    print("NUMBER OF CX-GATES: ", result["cx_count"], "\n")
    print("NUMBER OF T-GATES: ", result["t_count"], "\n")
    print("NUMBER OF T-DG-GATES: ", result["t_dg_count"], "\n")
    print("NUMBER OF P-GATES: ", result["p_count"], "\n")
    print("COMPLETE DATA: ", data, "\n")

    return result


def save_results_to_csv(results, filename="circuit_analysis.csv"):
    if not results:
        return

    keys = results[0].keys()

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def save_results_to_txt(results, filename="circuit_analysis.txt"):
    with open(filename, "w", encoding="utf-8") as out:
        for r in results:
            out.write(f"Number of Qubits: {r['num_qubits']}\n")
            out.write(f"Depth of circuit: {r['depth']}\n")
            out.write(f"Number of Gates: {r['gate_count']}\n")
            out.write(f"Number of x-Gates: {r['x_count']}\n")
            out.write(f"Number of Hadamard-Gates: {r['h_count']}\n")
            out.write(f"Number of u3-Gates: {r['u3_count']}\n")
            out.write(f"Number of cx-Gates: {r['cx_count']}\n")
            out.write(f"Number of t-Gates: {r['t_count']}\n")
            out.write(f"Number of t-dg-Gates: {r['t_dg_count']}\n")
            out.write(f"Number of p-Gates: {r['p_count']}\n")

            out.write("\n" + "#" * 50 + "\n\n")
