"""
DecodeLabs - AI Project 3: Tech Stack Recommender
Uses TF-IDF + Cosine Similarity (Content-Based Filtering)
"""

import csv
import math


def load_dataset(filepath):
    """Load job roles and their skills from CSV."""
    dataset = {}
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = row['job_role']
            skills = row['skills'].lower().split()
            dataset[role] = skills
    return dataset


def build_vocabulary(dataset):
    """Collect all unique skills across all job roles."""
    vocab = set()
    for skills in dataset.values():
        vocab.update(skills)
    return sorted(vocab)  # sorted for consistent indexing



def compute_tf(skill_list, vocab):
    """Term Frequency: how often a skill appears in a role's skill list."""
    tf = {}
    total = len(skill_list)
    for term in vocab:
        count = skill_list.count(term)
        tf[term] = count / total if total > 0 else 0
    return tf


def compute_idf(dataset, vocab):
    """Inverse Document Frequency: penalizes skills common across all roles."""
    N = len(dataset)
    idf = {}
    for term in vocab:
        docs_with_term = sum(1 for skills in dataset.values() if term in skills)
        # Add 1 to avoid division by zero
        idf[term] = math.log(N / (1 + docs_with_term))
    return idf


def compute_tfidf_vector(skill_list, vocab, idf):
    """Combine TF and IDF to produce a weighted feature vector."""
    tf = compute_tf(skill_list, vocab)
    vector = [tf[term] * idf[term] for term in vocab]
    return vector


def build_item_vectors(dataset, vocab, idf):
    """Build TF-IDF vectors for every job role in dataset."""
    vectors = {}
    for role, skills in dataset.items():
        vectors[role] = compute_tfidf_vector(skills, vocab, idf)
    return vectors


def build_user_vector(user_skills, vocab, idf):
    """
    Convert user's entered skills into a TF-IDF vector.
    Skills not in vocabulary are ignored (cold-start resilience).
    """
    normalized = [s.lower().replace(" ", "_") for s in user_skills]
    return compute_tfidf_vector(normalized, vocab, idf)



def cosine_similarity(vec_a, vec_b):
    """
    Cosine similarity = dot product / (magnitude A * magnitude B)
    Returns value between 0 (no match) and 1 (perfect match).
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0  # Cold start: no preferences entered

    return dot_product / (magnitude_a * magnitude_b)


def get_top_n_recommendations(user_vector, item_vectors, top_n=3):
    """
    Score every job role against user profile,
    sort descending, return Top-N results.
    """
    scores = {}
    for role, vector in item_vectors.items():
        scores[role] = cosine_similarity(user_vector, vector)

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked[:top_n]


def main():
    print("=" * 55)
    print("   DecodeLabs — Tech Stack Recommender (Project 3)")
    print("=" * 55)
    print("  Content-Based Filtering | TF-IDF + Cosine Similarity")
    print("-" * 55)

    # Load and prepare data
    dataset = load_dataset("raw_skills.csv")
    vocab = build_vocabulary(dataset)
    idf = compute_idf(dataset, vocab)
    item_vectors = build_item_vectors(dataset, vocab, idf)

    print(f"\n  Dataset loaded: {len(dataset)} job roles | {len(vocab)} unique skills\n")

    while True:
        print("Enter at least 3 skills (comma-separated).")
        print("Example: Python, Machine_Learning, SQL")
        print("Type 'quit' to exit.\n")

        user_input = input("  Your skills: ").strip()

        if user_input.lower() == 'quit':
            print("\n  Goodbye! Keep building. 🚀")
            break

        user_skills = [s.strip() for s in user_input.split(",")]

        # Validate minimum input
        if len(user_skills) < 3:
            print("\n  ⚠ Please enter at least 3 skills for accurate matching.\n")
            print("-" * 55)
            continue

        # Build user vector and get recommendations
        user_vector = build_user_vector(user_skills, vocab, idf)
        recommendations = get_top_n_recommendations(user_vector, item_vectors, top_n=3)

        # Check for cold start (all zeros)
        if all(score == 0 for _, score in recommendations):
            print("\n  ⚠ None of your skills matched our vocabulary.")
            print("  Try skills like: Python, SQL, AWS, Docker, Machine_Learning\n")
        else:
            print("\n  ✅ Top 3 Recommended Career Paths:")
            print("-" * 55)
            for rank, (role, score) in enumerate(recommendations, start=1):
                bar_len = int(score * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                print(f"  {rank}. {role:<30} {score:.4f}")
                print(f"     [{bar}]")

            print("-" * 55)
            print(f"  Skills entered: {', '.join(user_skills)}")

        print()

        again = input("  Try another profile? (y/n): ").strip().lower()
        if again != 'y':
            print("\n  Goodbye! Keep building. 🚀")
            break

        print("\n" + "=" * 55 + "\n")


if __name__ == "__main__":
    main()