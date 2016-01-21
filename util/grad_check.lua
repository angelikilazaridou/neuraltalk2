require 'util.progress_bar'

function numerical_gradient(J, theta)
    -- Initialize numgrad with zeros
   local numgrad = theta:clone():zero()
   local e = 1e-4
   
   local e_i = numgrad:clone()
   for i=1,theta:size()[1] do
      progress_bar.print(i, theta:size(1))
      e_i[i] = e
      numgrad[i] = (J(theta + e_i) - J(theta - e_i)) / (2*e)
      e_i[i] = 0
   end
   return numgrad
end

function grad_check(J, theta, callback)
   --Call callback before calling J
   local J_prev
   if callback then  J_prev = J; J = function(x) callback(); return J_prev(x) end end
   local num_grad = numerical_gradient(J, theta)
   local l, grad_params = J(theta)
   local err = (num_grad - grad_params):abs():sum() / math.max(num_grad:abs():sum(), grad_params:abs():sum())
   print(grad_params:cat(num_grad,2):cat(torch.cdiv(num_grad, grad_params), 2))
   return err
end
