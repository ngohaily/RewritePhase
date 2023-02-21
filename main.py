from rewrite_paragraph import GPTRewriteParrot
with open('input.txt', 'r') as file:
    content = file.read()
p = GPTRewriteParrot()
output = p.paraphrase(content)
with open('output.txt', 'w') as file:
    file.write(output)