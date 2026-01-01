import search
import csv
from pathlib import Path
import time


def process_results(query, result, metrics):

    gold_id = query["id"]
    found_ids = [r["chunk"]["id"] for r in result]  
    if gold_id in found_ids:
        metrics["top_5_hits"] += 1
        if found_ids[0] == gold_id:
            metrics["top_1_hits"] += 1


def log_results(index, query, result, metrics, log_path):
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        found_ids = [r["chunk"]["id"] for r in result]  
        writer.writerow([
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}",
            index,
            found_ids,
            query["id"],
            metrics["top_1_hits"],
            metrics["top_5_hits"]
        ])


def save_summary(index_dir, metrics):
    output_path = Path('output/summary.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(["Model", "Hit@1", "Hit@5"])
        writer.writerow([
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"{index_dir.name}",
            f"{metrics['top_1_accuracy']:.2f}",
            f"{metrics['top_5_accuracy']:.2f}"
        ])



def main():
    index_dir = Path("index_4")
    log_path = Path(f'output/results_{index_dir}.csv')
    log_path.parent.mkdir(parents=True, exist_ok=True)

    queries_1_overlap = [
        {"id": "book-000050", "text": "Cold draught entered from window sill"},
        {"id": "book-000982", "text": "Injury nearly pierced his groin, healed with extreme difficulty"},
        {"id": "book-000421", "text": "Part of crew survived exposure and reached land in boats"},
        {"id": "book-001133", "text": "Pequod hoisted sails and pursued Moby-Dick from leeward wake"},
        {"id": "book-001000", "text": "Artificial leg prepared to look real, ready for use tomorrow"},
        {"id": "book-000765", "text": "Tashtego planted iron in stricken whale"}
    ]

    queries_3_overlap= [
        {"id": "book-001127", "text": "Sun and whale died together at sunset"},
        {"id": "book-001179", "text": "Ahab resolves to discuss philosophies with Pip"},
        {"id": "book-001220", "text": "Moby Dick withdrew, thrusting head up and down"},
        {"id": "book-001096", "text": "Dying men shown indulgence despite consternation"},
        {"id": "book-000912", "text": "American fishermen created their own unwritten whaling laws"},
        {"id": "book-001011", "text": "English Captain refused to hunt White Whale again"},
        {"id": "book-001234", "text": "Ahab as keel directing unified crew toward fatal goal"},
        {"id": "book-001178", "text": "Sudden squall compared to Ahab’s fiery temperament"},
        {"id": "book-000471", "text": "Ships often lost men and crews during Pacific voyages"},
        {"id": "book-000668", "text": "Few successful darts despite many chances"},
    ]

    queries_index_4 = [
    {"id": "book-001068", "text": "Efficient in countless mechanical emergencies during long voyages"},
    {"id": "book-000785", "text": "Tashtego struggled inside whale head, sinking perilously deep"},
    {"id": "book-000834", "text": "Greeks and Romans also doubted whale-related traditions"},
    {"id": "book-000202", "text": "Criticized fasting practices like Ramadan as unhealthy and useless"},
    {"id": "book-000843", "text": "He balanced lance like a juggler before throwing"},
    {"id": "book-000412", "text": "His intellect remained intact despite deepening monomania"},
    {"id": "book-000899", "text": "Herd of whales recalled from fright, crowding together"},
    {"id": "book-001177", "text": "Carpenter prepared coffin as life-buoy for the ship"},
    {"id": "book-000369", "text": "Crew assembled, watching Ahab with uneasy curiosity"},
    {"id": "book-001125", "text": "Mates and harpooneers danced with Polynesian girls on deck"}
]


    current_queries = queries_index_4

    metrics: dict[str, float] = {
        "top_1_hits": 0,
        "top_5_hits": 0
    }

    for index, query in enumerate(current_queries):
        result = search.execute_query(query, index_dir, top_k=5)
        process_results(query, result, metrics)
        log_results(index, query, result, metrics, log_path)

    metrics["top_1_accuracy"] = metrics["top_1_hits"] / len(current_queries)
    metrics["top_5_accuracy"] = metrics["top_5_hits"] / len(current_queries)

    print(metrics)
    save_summary(index_dir, metrics)
    

if __name__ == "__main__":
    main()