from src.pipeline import hierarchical_processing_pipeline

problem_statement = "Reduce server costs for a cloud-based SaaS platform."
solutions = [
    "Use spot instances...",
    "Optimize auto-scaling...",
    "Migrate to ARM-based servers..."
]

output_report = hierarchical_processing_pipeline(problem_statement, solutions)
print("\n🔹 Generated Report:\n")
print(output_report)
