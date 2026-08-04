## **Introduction**

This project demonstrates the optimizations of Qdrant, a vector-first database, showcasing how to understand the search quality, balance search speed and accuracy, modify the index, and quantize the stored embeddings.

In the SOLUTION.md you find results and findings/observations for the different runs.

## **Project Structure**

Here are the main starter directories and files in this repo:

```
├── .github/
│   └── PULL_REQUEST_TEMPLATE.md
├── tasks/
│   ├── task_1.md
│   ├── task_2.md
│   ├── task_3.md
│   └── SOLUTION.md
├── dataset/
│   └── queries_embeddings.json
├── README.md
├── pyproject.toml
├── uv.lock
└── .gitignore
```

## **Setup**

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies.

* Install uv if you don't have it yet (see the [installation guide](https://docs.astral.sh/uv/getting-started/installation/))
* Install the project dependencies:
  ```
  uv sync
  ```
* Run your solution script for a task:
  ```
  uv run python <your_script>.py
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
