require('nn')
require('nngraph')
paths.dofile('LinearNB.lua')

local function build_memory(inputs, input_size, output_size, hops)
    local hid = {}   
    hid[0] = query

    local shareList = {}
    shareList[1] = {}

    local images = inputs[1]
    local query = inputs[2]
   
    --embed images in some multimodal space     
    local Iin_c = nn.Linear(input_size, output_size)(images)
    Iin_c = nn.ReLu(Iin_c)

    --embed again images in another space (do we need this?)
    local Iin_m = nn.Linear(input_size, output_size)(images)
    Iin_m = nn.ReLu(Iin_m)


    for h = 1, hops do
        local hid3dim = nn.View(1, -1):setNumInputDims(1)(hid[h-1])
        local MMaout = nn.MM(false, true):cuda()
        --dot product for similarity between hidden state and query
        local Aout = MMaout({hid3dim, Iin_c})
        local Aout2dim = nn.View(-1):setNumInputDims(2)(Aout)
        -- similarities to attention probabilities
        local P = nn.SoftMax()(Aout2dim)
        
        local probs3dim = nn.View(1, -1):setNumInputDims(1)(P)
        local MMbout = nn.MM(false, false):cuda()
        --weighted average of elements in memory
        local Bout = MMbout({probs3dim, Iin_m})

        --combine current attention with previous for hop
        local C = nn.LinearNB(output_size, output_size)(hid[h-1])
        table.insert(shareList[1], C)
        local D = nn.CAddTable()({C, Bout})
      
        --save current hop
        hid[h] = D

    end

    local outputs = {}
    table.insert(outputs,hid[#hid])
    table.insert(outputs,shareList)

    return outputs
end

function g_build_model(input_size, output_size, hops)
    
    --input
    local query = nn.Identity()()
    local images = nn.Identity()()

    table.insert(inputs, images)
    table.insert(inputs, query)
 
    local hid, shareList = build_memory(inputs, input_size, output_size, hops)
    --input to model is  {images, query} and output is the visual vector 
    local model = nn.gModule(inputs,hid[#hid])
    model:cuda()
    -- IMPORTANT! do weight sharing after model is in cuda
    for i = 1,#shareList do
        local m1 = shareList[i][1].data.module
        for j = 2,#shareList[i] do
            local m2 = shareList[i][j].data.module
            m2:share(m1,'weight','bias','gradWeight','gradBias')
        end
    end
    return model
end




