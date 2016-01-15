require 'nn'
require 'nngraph'

local MemNN = {}

function MemNN.build_memory(input_size, output_size, mem_size, hops)
    

   local inputs = {}
   -- for query
   local query = nn.Identity()()
   -- each query is of size batch_size x output size
   table.insert(inputs, query) 
   
   --for memories
   local mem_entries = {}
   for i=1,mem_size do
     --each entry is of size batch_size x input_size
       local entry = nn.Identity()()
       table.insert(inputs, entry) --insert mem entry one by one
       -- mem entry is of size batch_size x output_size
       local mem_entry = nn.ReLU()(nn.Linear(input_size, output_size)(entry)) --embedding entry in memory
       
       if i>1 then -- share parameters of each mem_entry with first one
          mem_entry.data.module:share(mem_entries[1].data.module,'weight','bias','gradWeight','gradBias')
       end
       table.insert(mem_entries, mem_entry)      
       
   end
   
  
   --create a tensor out of the table of size (mem_size x output_size)
   local all_mem_entries = nn.JoinTable(1)(mem_entries)
   --group the data in the tensor  by mem_size per batch  (mem_size x output_size)
   local mem_matrix = nn.View(#mem_entries,-1)(all_mem_entries)    
    --query has 1 x output_size  (1 x output_size)
   local  query_matrix= nn.View(1, -1)(query)
   --dot product for similarity between query and memories (1 x mem_size)
   local dot_product = nn.MM(false, true)
   -- dot product result (1 x mem_size)
   local sims = dot_product({query_matrix, mem_matrix})
   -- similarities to attention probabilities (1x mem_size)
   local probs = nn.SoftMax()(sims)
   --weighted average
   local weighted_average = nn.MM(false, false)
   -- (1 x mem_size
   local output = weighted_average({probs, mem_matrix}) 

   return nn.gModule(inputs, {output})
   
end
return MemNN
