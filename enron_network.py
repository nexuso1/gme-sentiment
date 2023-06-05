"""
This script handles the creation of a network from the Enron email dataset
"""

import networkx as nx
import os
import email.parser
import re

DATA_FOLDER = 'D:\\enron_data\\maildir'
NETWORK_PATH = '.\\Networks\\enron_network_new.gz'

def save_graph(graph, path) -> None:
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    nx.readwrite.write_gpickle(graph, path)

    print("Graph saved. Status: {} nodes, {} edges".format(graph.number_of_nodes(), graph.number_of_edges()))

def remove_spaces(string):
    return re.sub('[\n\t]', '', string)

def create_graph():
    graph = nx.DiGraph()
    parser = email.parser.HeaderParser()
    for path, dirnames, files in os.walk(DATA_FOLDER):
        for file in files:
            with open(os.path.join(path, file), 'r') as file:
                message = parser.parse(file)
            
            sender = remove_spaces(message['From'])
            recipients = []

            # To is None when the recipients are undisclosed
            if message['To'] is None:
                
                # It's some weird Calendar entry or something, can be skipped
                if message['X-to'] is None:
                    continue

                recipients = message['X-to'].split(', ')
            
            else:
                recipients = message['To'].split(', ')
            
            recipients = list(map(remove_spaces, recipients))
            for recipient in recipients:
                if not graph.has_edge(sender, recipient):
                    graph.add_edge(sender, recipient, weight=0)

                graph[sender][recipient]['weight'] += 1
            
            # Add Cc'd people
            if 'Cc' in message.items():
                cc_recipients = message['Cc']
                for recipient in list(map(remove_spaces, cc_recipients.split(', '))):

                    if not graph.has_edge(sender, recipient):
                        graph.add_edge(sender, recipient, weight=0)

                    graph[sender][recipient]['weight'] += 1

                
                # Add Bcc'd people, but they sometimes overlap so check for that
                if 'Bcc' in message.items():
                    for recipient in list(map(remove_spaces, message['Bcc'].split(', '))):
                        if recipient in cc_recipients:
                            continue

                        if not graph.has_edge(sender, recipient):
                            graph.add_edge(sender, recipient, weight=0)
                        
                        graph[sender][recipient]['weight'] += 1

                continue

            # No Cc but has Bcc
            if 'Bcc' in message.items():
                for recipient in list(map(remove_spaces, message['Bcc'].split(', '))):
                    if not graph.has_edge(sender, recipient):
                        graph.add_edge(sender, recipient, weight=0)
                    
                    graph[sender][recipient]['weight'] += 1
    
    print("Number of non-email nodes:")
    print(len([n for n in graph.nodes() if "@" not in n]))

    print("Number of edges with beginnings/ends in non-email nodes:")
    print(len([(u,v) for u,v in graph.edges if '@' not in u or '@' not in v]))

    graph.remove_nodes_from([n for n in graph.nodes() if '@' not in n])

    print("Number of 0 degree nodes:")
    print(len([node for node in graph.nodes if graph.degree(node) == 0]))

    graph.remove_nodes_from([node for node in graph.nodes() if graph.degree(node) == 0])
    save_graph(graph, NETWORK_PATH)

def main():
    create_graph() 

if __name__ == '__main__':
    main()