require 'nn'
require 'nngraph'

local LSTM = {}
function LSTM.lstm(image_size, mem_size, input_size, output_size, rnn_size, n, dropout)
  
  -- there will be 2*n+1 inputs + mem_size
  --RNN inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- input encoding word
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  -- each query is of size batch_size x rnn_size
  -- last layer of previous unit
  local query = inputs[2*n+1]
  
  --for memories
  local mem_entries = {}
  for i=1,mem_size do
    --each entry is of size batch_size x image_size
    local entry = nn.Identity()()
    table.insert(inputs, entry) --insert mem entry one by one
    -- mem entry is of size batch_size x rnn_size
    local mem_entry = nn.ReLU()(nn.Linear(image_size, rnn_size)(entry)) --embedding entry in memory

    if i>1 then -- share parameters of each mem_entry with first one
        mem_entry.data.module:share(mem_entries[1].data.module,'weight','bias','gradWeight','gradBias')
    end
    table.insert(mem_entries, mem_entry)
  end


  --create a tensor out of the table of size (mem_size x x batch_size x rnn_size)
  local all_mem_entries = nn.JoinTable(2)(mem_entries)
  --convert this into a 3D of size batch x mem_size x rnn_size
  local mem_entries_3D = nn.View(#mem_entries,-1):setNumInputDims(1)(all_mem_entries)
   --query has batch_size x rnn_size ->   (batch_size x rnn_size x 1)
  local query_3D= nn.View(rnn_size, -1):setNumInputDims(1)(query)
  --dot product for similarity between query and memories (1 x mem_size)
  local dot_product = nn.MM(false, false)
  -- dot product result (batch_size x mem_size x 1)
  local sims = dot_product({mem_entries_3D, query_3D})
  -- throw the dummy dimension 
  local sims_2D = nn.Select(3,1)(sims)
  -- similarities to attention probabilities (batch_size x mem_size)
  local probs = nn.SoftMax()(sims_2D)
  -- add dummy dimension to convert probs 2D tensor to 3D for MM 
  local probs_3D = nn.View(mem_size,-1):setNumInputDims(1)(probs)
  --weighted average
  local weighted_average = nn.MM(true, false)
  -- (batch_size x rnn_size x 1)
  local I = weighted_average({mem_entries_3D, probs_3D})
  --get rid of dummy dimension
  local Im = nn.Select(3,1)(I)

  local dropout = dropout or 0 

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    if L == 1 then
       local I2h = nn.Linear(rnn_size, 4 * rnn_size)(Im):annotate{name='I2h_'..L}
       i2h = nn.CAddTable()({i2h, I2h})
    end
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return LSTM

