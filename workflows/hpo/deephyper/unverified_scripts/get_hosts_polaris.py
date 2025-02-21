import sys

if __name__ == "__main__":
    ranks_per_node = 4
    fname = sys.argv[1]
    output = ""
    with open(fname, "r") as f:
        for i, line in enumerate(f):
            line = line.strip("\n")
            if i == 0:
                output += f"{line}"
            for _ in range(ranks_per_node):
                output += f",{line}"
    print(output)