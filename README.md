## **Overview**
This project demonstrates the optimizations of Qdrant, a vector-first database, showcasing how to understand the search quality, balance search speed and accuracy, modify the index, and quantize the stored embeddings.

## **Project Structure**

Here are the main starter directories and files in this repo:

```
├── tasks/
│   ├── evaluate_search.py   # Script to evaluate ANN vs exact k-NN search (Task 1)
│   ├── task_1.md            # Task 1 details: Catching up and evaluation
│   ├── task_2.md            # Task 2 details
│   └── task_3.md            # Task 3 details
├── dataset/
│   └── queries_embeddings.json # Test dataset with 100 queries and embeddings
├── README.md
└── .gitignore
```

## **Setup and Usage**

### **Prerequisites**
- **Docker**: For running the Qdrant instance.
- **Python 3.8+**: For running the evaluation script.
- **Qdrant**: A running Qdrant instance with an existing `arxiv_papers` collection.

### **Getting Started**
1. **Ensure Qdrant is running**:
   Make sure your Qdrant Docker container is active and accessible at `localhost:6333`.

2. **Set up the Virtual Environment**:
   ```bash
   cd tasks
   python3 -m venv further-steps-with-qdrant
   source further-steps-with-qdrant/bin/activate
   pip install -r requirements.txt
   ```

### **Running the Evaluation (Task 1)**
To calculate the average precision and compare search times:
```bash
python3 evaluate_search.py
```

Typical results for the `arxiv_papers` collection:
- **Average Precision@10**: ~1.0000
- **Average ANN query time**: ~20-25 ms
- **Average exact k-NN query time**: ~190-200 ms

## **Tasks**

This project is divided into various tasks that you need to complete. The tasks are located in the tasks folder of the repository. Each task includes all the necessary objectives, suggested development steps, expected outcomes, and useful resources.

## **Useful Resources**

Each task will contain a collection of resources that will be helpful for you as you solve the task. There are links to topics in Hyperskill, documentation, and other helpful tutorials that you and your team can use. You may not always need to use all the provided resources if you're already familiar with the concepts. In addition to the provided resources, you can always discuss with your teammates and experts.

## **The flow**
Fork → Clone → Branch → Implement → PR → Review

* Fork this repo to your own GitHub account
* Create a new branch for each task (e.g., task-1) if applicable (if there is any code that has to be implemented)
* Implement the solution based on the markdown descriptions
* Push the branch to the forked repo
* Create a Pull Request from the fork back to the main repo
* We will review the PR and provide feedback through GitHub