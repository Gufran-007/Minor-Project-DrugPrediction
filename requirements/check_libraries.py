libraries = [
    "pandas",
    "numpy",
    "torch",
    "sklearn",
    "nltk",
    "matplotlib",
    "seaborn",
    "flask"
]

print("Checking your libraries...\n")

all_good = True

for lib in libraries:
    try:
        __import__(lib)
        print(f"  ✅  {lib} is installed")
    except ImportError:
        print(f"  ❌  {lib} is NOT installed — needs to be installed")
        all_good = False

print()
if all_good:
    print("🎉 Everything is installed! You're ready to go.")
else:
    print("⚠️  Some libraries are missing. Scroll up to see which ones.")