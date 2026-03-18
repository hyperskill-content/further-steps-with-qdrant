## **Introduction**
This project demonstrates the optimizations of Qdrant, a vector-first database, showcasing how to understand the search quality, balance search speed and accuracy, modify the index, and quantize the stored embeddings.

## **Project Structure**

Here are the main starter directories and files in this repo:

```
├── tasks/
│   ├── task_1.md
│   ├── task_2.md
│   └── task_3.md
├── dataset/
├── README.md
└── .gitignore
```

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

## **How to Run**

1. Ensure you have Qdrant running (e.g., via Docker).
2. Activate your virtual environment:
   ```bash
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the evaluation script:
   ```bash
   python3 tasks/evaluate_quantization.py
   ```

### Example Output
```json
[
    {
        "rescore": true,
        "avg_precision": 0.9960000000000001,
        "avg_query_time_ms": 8.629827499389648
    },
    {
        "rescore": false,
        "avg_precision": 0.8359999999999995,
        "avg_query_time_ms": 8.576357364654541
    }
]
```