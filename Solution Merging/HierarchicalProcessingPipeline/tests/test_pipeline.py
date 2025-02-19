from src.pipeline import hierarchical_processing_pipeline

def test_pipeline():
    """Tests pipeline with edge cases."""
    problem_statement = "Reduce server costs for a cloud-based SaaS platform."
    
    solutions = [
        "Use spot instances...",
        "Optimize auto-scaling...",
        "Migrate to ARM-based servers..."
    ]

    report = hierarchical_processing_pipeline(problem_statement, solutions)
    
    assert isinstance(report, str)
    assert len(report) > 0
    print("Test Passed!")

# Run test
if __name__ == "__main__":
    test_pipeline()
