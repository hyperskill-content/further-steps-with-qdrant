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

## **Observations**
=======================================================
hnsw_ef      avg_precision        avg_query_time_ms
=======================================================
10           0.8190               3.9521
20           0.8310               3.1071
50           0.8320               3.3232
100          0.8320               3.9691
200          0.8320               4.4074
=======================================================