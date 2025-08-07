import torch
from importlib.metadata import version

from models.Attentions import CausalAttention, MultiHeadAttention, MultiHeadAttentionWrapper, SelfAttention_v1, SelfAttention_v2

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
# print('Quick verification that the values in each row indeed sum to 1:')
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# print("Row 2 sum:", row_2_sum)

# print("All row sums:", attn_weights.sum(dim=-1))

# print("Previous 2nd context vector:", context_vec_2)

# 3.4 Implementing self-attention with trainable weights
batch = torch.stack((inputs, inputs), dim=0)

x_2 = inputs[1] # second input element
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size, d=2

print('Implementing a compact SelfAttention class')
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))


# 3.5 Hiding future words with causal attention

# 3.5.2 Masking additional attention weights with dropout

# 3.5.3 Implementing a compact causal self-attention class
torch.manual_seed(123)
print('CausalAttention')
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)

context_vecs = ca(batch)

print(context_vecs)
print("CausalAttention - context_vecs.shape:", context_vecs.shape)

# 3.6 Extending single-head attention to multi-head attention
torch.manual_seed(123)

context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
print('MultiHeadAttentionWrapper')
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print(context_vecs)
print("MultiHeadAttentionWrapper - context_vecs.shape:", context_vecs.shape)

torch.manual_seed(123)

print('MultiHeadAttention')
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("MultiHeadAttention - context_vecs.shape:", context_vecs.shape)