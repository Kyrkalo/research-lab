import torch
from importlib.metadata import version

print("torch version:", version("torch"))


# 3.3 Attending to different parts of the input with self-attention
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]  # 2nd input token is the query
# print(f'query: {query}')
attn_scores_2 = torch.empty(inputs.shape[0])
# print(f'scores: {attn_scores_2}')
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
    # print(f'   x_i: {x_i}; score: {i};')

# print(attn_scores_2)

# res = 0.
# print('')
# for idx, element in enumerate(inputs[0]):
#     print(f'idx: {idx}; query[idx]: {query[idx]}; inputs[0][idx]: {inputs[0][idx]};')
#     res += inputs[0][idx] * query[idx]
# print('')
# print(res)
# print(torch.dot(inputs[0], query))

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

query = inputs[1] # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)

#3.3.2 Computing attention weights for all input tokens
#Apply previous step 1 to all pairwise elements to compute the unnormalized attention score matrix:
attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

# We can achieve the same as above more efficiently via matrix multiplication:
attn_scores = inputs @ inputs.T
print(attn_scores)

# Similar to step 2 previously, we normalize each row so that the values in each row sum to 1:
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# Quick verification that the values in each row indeed sum to 1:
print('Quick verification that the values in each row indeed sum to 1:')
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)

print("All row sums:", attn_weights.sum(dim=-1))