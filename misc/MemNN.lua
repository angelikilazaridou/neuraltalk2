require 'nn'
require 'nngraph'

local MemNN = {}

function MemNN.build_memory(input_size, output_size, mem_size, hops)
    
   
   local inputs = {}
   
   local mem_entries = {}
   -- each entry is of size batch_size x input_size
   for i=1,mem_size do
       local entry = nn.Identity()()
       table.insert(inputs, entry) --insert mem entry one by one
       
       local mem_entry = nn.ReLU()(nn.Linear(input_size, output_size)(entry)) --embedding entry in memory
       if i>1 then -- share parameters of each mem_entry with first one
          mem_entry.data.module:share(mem_entries[1].data.module,'weight','bias','gradWeight','gradBias')
       end
       table.insert(mem_entries, mem_entry)      
   end
   
   local query = nn.Identity()()
   table.insert(inputs, query) -- for query
   

   
   local all_mem_entries = nn.JoinTable(1)(mem_entries)
   local mem_matrix=nn.View(#mem_entries,-1)(all_mem_entries)    
    --query has batch-size x output_size
   local  query_matrix= nn.View(1, -1):setNumInputDims(1)(query)
   --dot product for similarity between memories and query
   local dot_product = nn.MM(false, true)
   local sims = dot_product({query_matrix, mem_matrix})
   local sims_matrix = nn.View(-1):setNumInputDims(2)(sims)
   -- similarities to attention probabilities
   local probs = nn.SoftMax()(sims_matrix)
   --weighted average
   local probs_matrix = nn.View(1, -1):setNumInputDims(1)(probs)
   local weighted_average = nn.MM(false, false)
   local output = weighted_average({probs_matrix, mem_matrix}) 

   return nn.gModule(inputs, {output})
end
return MemNN
