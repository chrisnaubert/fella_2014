# Julia implementation of consav
# Interpolation for dimensions 1:3

function binary_search(init,xq,g,Ng)
    if xq<=g[1]
        return 1
    elseif xq>=g[Ng-1]
        return Ng-1
    else
        itrs=0
        low=init
        high=Ng
        while itrs<1000
            id = floor(Int64,(low+high)/2)
            dis=xq-g[id]
            if dis<=(g[id+1]-g[id]) && dis>=0.0
                return id
            elseif dis>=(g[id-1]-g[id]) && dis<=0.0 
                return id-1
            elseif dis>(g[id+1]-g[id]) && dis>0.0
                low=id+1
            elseif dis<(g[id-1]-g[id]) && dis<0.0
                high=id-1
            end
            itrs+=1
        end
        return 0
    end
end


function binary_search_con(init,xq,g,Ng)
    if xq<=g[1]
        return 0
    elseif xq>=g[Ng-1]
        return Ng-1
    else
        itrs=0
        low=init
        high=Ng
        while itrs<1000
            id = floor(Int64,(low+high)/2)
            dis=xq-g[id]
            if dis<=(g[id+1]-g[id]) && dis>=0.0
                return id
            elseif dis>=(g[id-1]-g[id]) && dis<=0.0 
                return id-1
            elseif dis>(g[id+1]-g[id]) && dis>0.0
                low=id+1
            elseif dis<(g[id-1]-g[id]) && dis<0.0
                high=id-1
            end
            itrs+=1
        end
        return 0
    end
end


function interp_3d(g1,g2,g3,xq1,xq2,xq3,yq,Ng1,Ng2,Ng3)
    idx_1=binary_search(1,xq1,g1,Ng1)
    idx_2=binary_search(1,xq2,g2,Ng2)
    idx_3=binary_search(1,xq3,g3,Ng3)
    return interp_3d_given_index(g1,g2,g3,xq1,xq2,xq3,yq,idx_1,idx_2,idx_3)
end

function interp_3d_given_index(g1,g2,g3,xq1,xq2,xq3,yq,idx_1,idx_2,idx_3)
    deno=(g1[idx_1+1]-g1[idx_1])*(g2[idx_2+1]-g2[idx_2])*(g3[idx_3+1]-g3[idx_3])

    num_1_L=g1[idx_1+1]-xq1
    num_1_R=xq1-g1[idx_1]
    num_2_L=g2[idx_1+1]-xq1
    num_2_R=xq2-g2[idx_2]
    num_3_L=g3[idx_3+1]-xq3
    num_3_R=xq3-g3[idx_3]
    
    numer=0.0
    for j1 ∈ 1:2
        num_1=j1==1 ? num_1_L : num_1_R
        for j2 ∈ 1:2
            num_2=j2==1 ? num_2_L : num_2_R
            for j3 ∈ 1:2
                num_3=j3==1 ? num_3_L : num_3_R

                numer+=num_1*num_2*num_3*yq[idx_1+(1-j1),idx_2+(1-j2),idx_3+(1-j3)]
            end
        end
    end
    return numer/deno    
end



function interp_2d(g1,g2,xq1,xq2,yq,Ng1,Ng2)
    idx_1=binary_search(1,xq1,g1,Ng1)
    idx_2=binary_search(1,xq2,g2,Ng2)
    return interp_2d_given_index(g1,g2,xq1,xq2,yq,idx_1,idx_2)
end

function interp_2d_given_index(g1,g2,xq1,xq2,yq,idx_1,idx_2)
    deno=(g1[idx_1+1]-g1[idx_1])*(g2[idx_2+1]-g2[idx_2])
    num_1_L=g1[idx_1+1]-xq1
    num_1_R=xq1-g1[idx_1]
    num_2_L=g2[idx_2+1]-xq2
    num_2_R=xq2-g2[idx_2]
    
    numer=0.0
    for j1 ∈ 1:2
        num_1=j1==1 ? num_1_L : num_1_R
        for j2 ∈ 1:2
            num_2=j2==1 ? num_2_L : num_2_R
            numer+=num_1*num_2*yq[idx_1+(j1-1),idx_2+(j2-1)]
            
        end
    end
    #@printf("Numer is %.8f, deno is %.8f, ratio is %.8f",numer,deno,numer/deno)
    return numer/deno    
end


function interp_1d(g,xq,yq,Ng)
    idx=binary_search(1,xq,g,Ng)
    return interp_1d_given_index(g,xq,yq,idx)
end

function interp_1d_given_index(g,xq,yq,idx)
    x_L=g[idx]
    x_H=g[idx+1]
    y_L=yq[idx]
    y_H=yq[idx+1]
    return interp_1d_given_vals(x_L,x_H,y_L,y_H,xq)  
end

function interp_1d_given_vals(x_L,x_H,y_L,y_H,xq)
    return ((xq-x_L)*y_H+(x_H-xq)*y_L)/(x_H-x_L)    
end

function extrap_if_inf_1d(g,xq,yq,Ng)
    idx=binary_search(1,xq,g,Ng)
    return extrap_if_inf_1d_given_index(g,xq,yq,idx+1)
end

function extrap_if_inf_1d_given_index(g,xq,yq,idx)
    idx_x=idx
    if yq[idx]==Inf
        idx_x=idx_x+1
    elseif yq[idx+1]==Inf
        idx_x=idx_x-1
    end
    x_L=g[idx_x]
    x_H=g[idx_x+1]
    y_L=yq[idx_x]
    y_H=yq[idx_x+1]
    return interp_1d_given_vals(x_L,x_H,y_L,y_H,xq)  
end