import requests

def fetch(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        return response.json()  # Return the JSON data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None
    
def getRideStatuses():
    waitTimes = fetch("https://queue-times.com/parks/16/queue_times.json") # Disneyland has id 16

    statuses = {}

    for attraction in waitTimes["lands"]:
        for ride in attraction["rides"]:
            name = ride["name"]
            is_open = ride["is_open"]
            wait_time = ride["wait_time"]
            statuses[name] = (is_open, wait_time)

    return statuses