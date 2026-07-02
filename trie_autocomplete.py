# trie_autocomplete.py

class TrieNode:
    def __init__(self):
        # A dictionary mapping characters to their child nodes
        # Space complexity is highly optimized as prefixes are shared
        self.children = {}
        self.is_end_of_word = False

class PrefixTree:
    def __init__(self):
        """
        Initializes the root of the Trie.
        The root node contains no character itself, just links to the first letters.
        """
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        Time Complexity: O(L) where L is the length of the word.
        """
        current_node = self.root
        for char in word:
            # If the character path doesn't exist, create it
            if char not in current_node.children:
                current_node.children[char] = TrieNode()
            # Traverse down the tree
            current_node = current_node.children[char]
        
        # Mark the final node as a valid word boundary
        current_node.is_end_of_word = True

    def _find_node(self, prefix: str) -> TrieNode:
        """Helper method to traverse down to the node representing the prefix."""
        current = self.root
        for char in prefix:
            if char not in current.children:
                return None
            current = current.children[char]
        return current

    def get_autocomplete_suggestions(self, prefix: str) -> list[str]:
        """
        Returns all valid words in the database that start with the given prefix.
        """
        start_node = self._find_node(prefix)
        if not start_node:
            return [] # Prefix doesn't exist in our dictionary

        suggestions = []
        
        # Depth-First Search (DFS) to find all completed words branching from this node
        def dfs(node, current_path):
            if node.is_end_of_word:
                suggestions.append(current_path)
                
            for char, child_node in node.children.items():
                dfs(child_node, current_path + char)
                
        dfs(start_node, prefix)
        return suggestions


# --- Quick Test ---
if __name__ == "__main__":
    print("🌳 Booting Prefix Tree (Trie) Engine...")
    trie = PrefixTree()
    
    # Simulating a technical search database
    tech_words = [
        "machine", "macbook", "macro", "machine learning",
        "kubernetes", "kafka", "kaggle", "keras",
        "tensor", "tensorflow", "tensorrt"
    ]
    
    print(f"📥 Loading {len(tech_words)} terms into the Trie...")
    for word in tech_words:
        trie.insert(word)
        
    # Test 1: Autocomplete 'mac'
    prefix_1 = "mac"
    print(f"\n🔍 Searching for prefix: '{prefix_1}'")
    print(f"   Suggestions: {trie.get_autocomplete_suggestions(prefix_1)}")
    
    # Test 2: Autocomplete 'tensor'
    prefix_2 = "tensor"
    print(f"\n🔍 Searching for prefix: '{prefix_2}'")
    print(f"   Suggestions: {trie.get_autocomplete_suggestions(prefix_2)}")