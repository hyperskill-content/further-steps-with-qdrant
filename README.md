## **Introduction**
This project demonstrates the optimizations of Qdrant, a vector-first database, showcasing how to understand the search quality, balance search speed and accuracy, modify the index, and quantize the stored embeddings.

## **Project Structure**

Here are the main starter directories and files in this repo:

```
├── tasks/
│   ├── evaluate_hnsw_ef.py
│   ├── task_1.md
│   ├── task_2.md
│   └── task_3.md
├── dataset/
├── requirements.txt
├── README.md
└── .gitignore
```

## **Installation**

To set up the virtual environment and install the required dependencies, run:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## **Usage**

### **Evaluating HNSW ef Parameter**

To evaluate the impact of the `hnsw_ef` parameter on search precision and performance, run:

```bash
cd tasks
python3 evaluate_hnsw_ef.py
```

This script will:
* Load queries and embeddings from `dataset/queries_embeddings.json`.
* Establish ground truth using exact search (`exact=True`).
* Test various `hnsw_ef` values (10, 20, 50, 100, 200).
* Output average precision and average query time for each value.

Typical results for the `arxiv_papers` collection:
- `hnsw_ef = 10`: Average Precision ~0.94, Query time ~4.7 ms
- `hnsw_ef = 20`: Average Precision ~0.98, Query time ~5.1 ms
- `hnsw_ef = 50`: Average Precision ~1.00, Query time ~6.1 ms
- `hnsw_ef = 100`: Average Precision ~1.00, Query time ~7.8 ms
- `hnsw_ef = 200`: Average Precision ~1.00, Query time ~10.9 ms

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