# Nova Evaluation System

A simple evaluation system for testing Nova's AI assistant performance.

## Table of Contents
- [Quick Start](#quick-start-5-minutes)
- [What's in this folder?](#whats-in-this-folder)
- [Which command should I use?](#which-command-should-i-use)
- [Understanding Results](#understanding-results)
- [Customization Guide](#customization-guide)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Quick Start (5 minutes)

### Step 1: Setup (One-time)
```bash
# Navigate to project root
cd /home/work/work_liam/ds-ai-aiassistant-core-api

# Install dependencies
uv sync --dev

# Optional: Enable short commands (nova-eval, nova-test)
uv pip install -e .
```

### Step 2: Get Credentials
Ask your senior developer for AWS credentials, then set them:
```bash
export AWS_ACCESS_KEY_ID="ASIA..."
export AWS_SECRET_ACCESS_KEY="8Y7G..."
export AWS_SESSION_TOKEN="IQoJ..."
```

### Step 3: Run Your First Test
```bash
# Quick test (30 seconds, 1 question)
uv run nova-test

# Full evaluation (20 minutes, 42 questions)
uv run nova-eval
```

**That's it!** You should see evaluation scores like `relevance: 0.95, accuracy: 0.80`.

---

## What's in this folder?

```
evaluation/
├── README.md                      # You are here!
├── configuration/
│   ├── experiment.toml            # What to test & settings
│   ├── evaluator_prompts.yaml     # Evaluation criteria prompts
│   └── general.toml               # How to score answers
├── eval_experiment_runner.py      # MAIN: Full 42-question evaluation
├── stand_alone_langfuse_test.py   # TEST: Quick 1-question test
├── eval_lib.py                    # Core functions (don't run directly)
└── LANGFUSE_INTEGRATION_GUIDE.md  # Detailed docs on Langfuse integration
```

---

## Which command should I use?

| Purpose | Command | Time | Questions |
|---------|---------|------|-----------|
| **Learning/Debugging** | `uv run nova-test` | 30s | 1 |
| **Production Testing** | `uv run nova-eval` | 20min | 42 |

### Alternative Commands (if short commands don't work)
```bash
# Instead of nova-test
uv run python evaluation/stand_alone_langfuse_test.py

# Instead of nova-eval
uv run python evaluation/eval_experiment_runner.py
```

---

## Understanding Results

### What Gets Measured
Every Nova answer is scored on 3 key criteria (0.0 = bad, 1.0 = perfect):

| Score | What it measures | Good score | Bad example |
|-------|------------------|------------|-------------|
| **Retrieval Relevance** | Are retrieved docs relevant to question? | > 0.85 | Retrieves billing docs when asked about WiFi |
| **Accuracy** | Is the answer factually correct? | > 0.80 | Says "€50/month" when it's actually €30 |
| **Hallucination** | Does answer stay true to sources? | < 0.30 | Makes up features not in retrieved docs |

**Note:** Lower hallucination scores are better (0.0 = no hallucinations)

### Reading Results

**Good Results (System working well):**
```
Results: 38/42 successful (90.5%)
EVALUATION SCORES:
   retrieval_relevance : 0.920 (n=38)  (Finding right documents)
   accuracy            : 0.845 (n=38)  (Mostly correct info)
   hallucination       : 0.150 (n=38)  (Minimal made-up content)
   Average            : 0.805           (Overall strong performance)
```

**Concerning Results (Needs investigation):**
```
Results: 25/42 successful (59.5%)  (Low success rate!)
EVALUATION SCORES:
   retrieval_relevance : 0.450 (n=25)  (Wrong documents retrieved)
   accuracy            : 0.420 (n=25)  (Wrong information!)
   hallucination       : 0.680 (n=25)  (Making up too much!)
   Average            : 0.517           (System needs fixes)
```

### Where to See Detailed Results
After running evaluations, check: **https://langfuse.dev01.datascience-tmnl.nl**
- Look for your experiment run (e.g., `nova_experiment_v1_20250827_161403`)
- Click into individual questions to see exactly what went wrong

---

## Customization Guide

### Quick Customizations

#### Test Different Questions
```bash
# Edit this file to change which questions are tested:
nano configuration/experiment.toml

# Change the dataset:
dataset_name = "nova_clean_dataset_v1"      # Default: 42 questions
dataset_name = "nova_test_dataset"          # Smaller test set
dataset_name = "your_custom_dataset"        # Your own questions
```

#### Test Only Specific Things
```bash
# Edit configuration/experiment.toml
# Change this section to test only what you care about:
evaluations = [
    "retrieval_relevance",  # Are retrieved docs relevant?
    "accuracy",            # Is information correct?
    # "hallucination",     # Comment out = skip this test
]
```

#### Change Experiment Name
```bash
# Edit configuration/experiment.toml
experiment_name = "my_custom_test"          # Your personal test
experiment_name = "nova_experiment_v2"      # New version test
```

### Custom Question Sets

#### Method 1: JSON File (Easiest for developers)
```bash
# 1. Create evaluation/my_questions.json:
[
  {
    "question": "Hoe kan ik mijn factuur downloaden?",
    "expected_answer": "Log in op Mijn Odido en ga naar factuursectie",
    "intent": "factuur uitleg",
    "knowledge_base": "thuis_sf"
  },
  {
    "question": "Wat kost roamen in Europa?",
    "expected_answer": "EU roaming is gratis binnen je bundel",
    "intent": "mobiel",
    "knowledge_base": "mobiel_community"
  }
]

# 2. Edit eval_experiment_runner.py (around line 325), uncomment:
# dataset_source="json",
# json_filename="my_questions.json"

# 3. Run: uv run nova-eval
```

#### Method 2: CSV File (Easiest for non-developers)
```bash
# 1. Create evaluation/my_questions.csv:
question,expected_answer,intent,knowledge_base
"Hoe zeg ik mijn abonnement op?","Via Mijn Odido of bel klantenservice","mobiel","mobiel_sf"
"Wat kost roamen in Europa?","EU roaming is gratis binnen je bundel","mobiel","mobiel_community"

# 2. Edit eval_experiment_runner.py, uncomment CSV section around line 258
# 3. Run: uv run nova-eval
```

#### Method 3: Add to Langfuse Programmatically
```bash
# Add example questions
uv run python evaluation/add_questions_to_langfuse.py --example

# Import from JSON file
uv run python evaluation/add_questions_to_langfuse.py --json-file my_questions.json

# Specify custom dataset
uv run python evaluation/add_questions_to_langfuse.py --dataset my_custom_dataset

# Then run evaluation (picks up new questions automatically)
uv run nova-eval
```

#### Method 4: Add via Langfuse UI (Backup option)
```bash
# 1. Go to: https://langfuse.dev01.datascience-tmnl.nl
# 2. Navigate to "Datasets" → "nova_clean_dataset_v1"
# 3. Click "Add Item"
# 4. Fill in question, expected answer, metadata
# 5. Run: uv run nova-eval
```

---

## Advanced Usage

### Different Experiment Types

#### Quality Gate Testing (Before releases)
```bash
# Test all 42 questions with strict thresholds
uv run nova-eval

# Check results meet minimum requirements:
# Relevance > 0.85, Accuracy > 0.80, Helpfulness > 0.85
```

#### Performance Monitoring (Weekly checks)
```bash
# Run same test weekly, compare trends:
experiment_name = "weekly_monitoring_2025_01_15"  # Edit experiment.toml
uv run nova-eval

# Compare with previous weeks in Langfuse dashboard
```


#### Debug Specific Issues
```bash
# If customers report "Nova doesn't understand":
evaluations = ["retrieval_relevance"]  # Focus on document retrieval

# If customers report "Wrong information":
evaluations = ["accuracy", "hallucination"]  # Focus on correctness

# Edit configuration/experiment.toml, then run:
uv run nova-eval
```

### Advanced Configuration Options

#### Timeout & Retry Settings
```toml
# In configuration/experiment.toml
[parameters]
timeout = 60              # Wait longer for complex questions
retry_attempts = 5        # More retries for flaky connections
batch_size = 5           # Slower but more stable
```

#### Target Score Thresholds
```toml
# In configuration/experiment.toml
[improvements]
target_relevance = 0.95        # Stricter relevance requirement
target_accuracy = 0.75         # Lower accuracy threshold for testing
target_helpfulness = 1.0       # Perfect helpfulness required
```

#### Environment Settings
```toml
[parameters]
mode = "prod"            # Use production Nova instance
mode = "dev"             # Use development Nova instance (default)
```

---

## How Credentials Work

### Hybrid Credential System
The evaluation system uses a **hybrid credential approach** that works in both production and local environments:

1. **Default AWS Credential Chain (Production/CI/ECS)**
   - Tries IAM roles, instance profiles, or container credentials first
   - No manual setup needed in production environments
   - Automatically works in GitHub Actions, ECS tasks, or EC2 instances

2. **Manual Credentials Fallback (Local Development)**
   - Falls back to environment variables if default chain fails
   - Uses `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
   - Perfect for local development with temporary credentials

### Credentials Flow
```
Try Default Chain → Success? → Use it
      ↓ Fail?
Try Environment Vars → Success? → Use it
      ↓ Fail?
Show Error
```

This ensures the code works everywhere without modification!

---

## Troubleshooting

### Common Problems & Solutions

#### "Unable to locate credentials"
```bash
# Problem: AWS credentials not set
# Solution: Get fresh credentials from your senior developer:
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_SESSION_TOKEN="your_token"
```

#### "Failed to spawn: nova-test"
```bash
# Problem: Short commands not installed
# Solution: Install the package:
uv pip install -e .

# Alternative: Use longer commands:
uv run python evaluation/stand_alone_langfuse_test.py
```

#### "Template not found for [evaluation_name]"
```bash
# Problem: Missing evaluation templates
# Solution: Check YAML configuration file:
cat configuration/evaluator_prompts.yaml | grep -A1 evaluators

# Make sure it has evaluators for: retrieval_relevance, accuracy, hallucination
```

#### "Connection to Nova WebSocket failed"
```bash
# Problem: Nova service might be down
# Solution: Ask your senior developer about Nova service status
# Check: https://status.datascience-tmnl.nl (if exists)
```

#### Low evaluation scores consistently
```bash
# Problem: System performance issues
# Investigation steps:

# 1. Check specific failing evaluations:
uv run nova-test  # Quick test to see which scores are low

# 2. Look at individual question details in Langfuse dashboard

# 3. Focus on specific problem areas:
# - accuracy < 0.5 → Nova giving wrong information
# - retrieval_relevance < 0.5 → Nova retrieving wrong documents
# - hallucination > 0.5 → Nova making up information

# 4. Run targeted tests:
# Edit configuration/experiment.toml:
evaluations = ["retrieval_relevance"]  # Test only the problem area
uv run nova-eval
```

### Getting Help

1. **Quick issues:** Check troubleshooting above
2. **Configuration problems:** Look at example files in this README
3. **AWS/Credentials:** Ask your senior developer for fresh credentials
4. **Understanding results:** Use Langfuse dashboard to see detailed traces
5. **Still stuck:** Ask your team - this system is actively maintained!

---

## Example Workflows

### Daily Development Workflow
```bash
# 1. Quick health check before starting work
uv run nova-test
# Expected: All scores > 0.8, takes 30 seconds

# 2. Make changes to Nova
# ... your development work ...

# 3. Test your changes
uv run nova-test
# Check: Did scores improve/stay the same?

# 4. Before committing major changes
uv run nova-eval
# Full test to make sure nothing broke
```

### Release Testing Workflow
```bash
# 1. Full evaluation on release candidate
experiment_name = "release_candidate_v2_1_0"  # Edit experiment.toml
uv run nova-eval

# 2. Check quality gates:
# Success rate > 90%
# Average scores meet targets (see "Understanding Results" above)
# No regressions vs previous release

# 3. If quality gates pass → approve release
# 4. If not → investigate failing questions in Langfuse dashboard
```

### Debugging Customer Issues Workflow
```bash
# Customer reports: "Nova gives wrong pricing information"

# 1. Focus test on accuracy
# Edit configuration/experiment.toml:
evaluations = ["accuracy", "relevance"]

# 2. Run focused evaluation
uv run nova-eval

# 3. Check results:
# - accuracy < 0.7 → Confirmed issue
# - Look for pricing-related questions in Langfuse dashboard
# - Find specific wrong answers and root causes

# 4. Fix issues, re-test
uv run nova-eval
# Confirm: accuracy improved
```

---

## Key Features & Improvements

### What Makes This System Robust

1. **Source Extraction Works**
   - Both `eval_lib.py` handlers properly extract sources from Nova's responses
   - Sources appear in Langfuse metadata for debugging
   - Robust parsing handles both dict and string payload formats

2. **External Evaluation Pattern**
   - `stand_alone_langfuse_test.py` uses self-contained evaluation
   - No Nova module imports needed
   - Direct Azure OpenAI calls for scoring
   - Works anywhere without Nova's virtual environment

3. **Hybrid Credentials**
   - Automatically uses IAM roles in production/CI
   - Falls back to manual credentials for local development
   - Same code works everywhere without modification

4. **Comprehensive Documentation**
   - This README for quick start and common tasks
   - `LANGFUSE_INTEGRATION_GUIDE.md` for deep technical details
   - Clear examples and troubleshooting steps

---

## Related Resources

- **Langfuse Dashboard:** https://langfuse.dev01.datascience-tmnl.nl
- **Nova Main Code:** `../src/` folder
- **Project README:** `../README.md`
- **Configuration Files:** `configuration/experiment.toml` and `configuration/evaluator_prompts.yaml`
- **Integration Guide:** `LANGFUSE_INTEGRATION_GUIDE.md`

---

## Tips for Junior Developers

### Understanding the Evaluation Flow
1. **Question** → Nova WebSocket → **Answer** (Nova processes the question)
2. **Answer** + **Question** → AI Evaluator → **Scores** (AI judges the quality)
3. **Scores** → Langfuse → **Dashboard** (Results stored and visualized)

### Learning Path
1. **Start simple:** `uv run nova-test` to see how it works
2. **Explore results:** Look at Langfuse dashboard to understand scoring
3. **Try customization:** Change a question in experiment.toml and re-run
4. **Run full test:** `uv run nova-eval` when you're comfortable
5. **Advanced usage:** Custom datasets, A/B testing, debugging specific issues

### Best Practices
- **Always test first** with `nova-test` before running full evaluations
- **Check Langfuse dashboard** to understand why scores are low/high
- **Use meaningful experiment names** so you can find results later
- **Document your changes** when you modify configuration files
- **Don't run full evaluations constantly** - they take 20 minutes and use resources
- **Don't ignore low scores** - they indicate real problems that affect customers

Happy evaluating!
