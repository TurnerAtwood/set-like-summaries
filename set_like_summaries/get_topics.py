import json

IN_FILE = './static/txt/ProjectData.txt'
OUT_FILE = './static/json/topics.json'
def main():
  topics = []

  with open(IN_FILE, 'r') as in_file:
    data = json.load(in_file)

    for topic in data:
      value = int(topic[4:])
      option = data[topic]['theme']
      topics.append((value, option))

  with open(OUT_FILE, 'w') as out_file:
    sorted_topics = sorted(topics)
    json.dump(sorted_topics, out_file)


if __name__ == '__main__':
  main()
