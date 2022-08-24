using QuantEcon: rouwenhorst
using QuantEcon: tauchen
using NLsolve
using LinearAlgebra: dot
import Printf.@printf
using Statistics: mean
using BSON: @save
# USE DIFFERENT NUMBER OF POINTS FOR AP AND A 
# Local files
include("interp_tools.jl")


mutable struct FirmParamsM
    alpha::Float64   
    beta::Float64 
    delta::Float64    
    rho::Float64 
    sigma::Float64
    rho_t::Float64 
    sigma_t::Float64
    theta::Float64
    kappa::Float64
    gamma::Float64
    zeta::Float64
    phi::Float64
    r::Float64
    amin::Float64
    amax::Float64
    hmin::Float64
    hmax::Float64
    cmin::Float64
end


function set_FirmParamsM()
    alpha= 0.4;     
    beta= 0.93;    
    delta= 0.02;     
    rho= 0.977;    
    sigma= 0.024;
    rho_t= 0.0;    
    sigma_t= 0.063;
    theta=0.77;
    kappa=0.075;
    gamma=0.0;
    zeta=0.80;
    phi=0.006
    r=0.06;

    amin=0.0
    amax=25.0
    hmin=1.0e-2
    hmax=10.0
    cmin=10e-3
    return FirmParamsM(alpha,beta,delta,rho,sigma,rho_t,sigma_t,theta,kappa,gamma,zeta,phi,r,amin,amax,hmin,hmax,cmin)
end


mutable struct DecisionMatricies{T<:Union{Float32,Float64}}
    con::Array{T,4}
    sav::Array{T,4}
    val_choice::Array{T,4}
    
    
    z_exo::Array{T,4}
    exp_val::Vector{T}
    exp_dval::Vector{T}

    W::Array{T,3}
    Wa::Array{T,3}
    V::Array{T,3}
    Va::Array{T,3}

    mask::Vector{Bool}

    c_node::Vector{T}
    z_node::Vector{T}
    val_node::Vector{T}
    ap_node::Vector{Int64}

    c_tmp::Vector{T}
    z_tmp::Vector{T}

    c_out::Vector{T}
    v_out::Vector{T}


    V_old::Array{T,3}
    segment::Array{Int64,3}
end

function set_DecisionMatricies(gd)
    con=zeros(gd.N_ghp,gd.N_gap,gd.N_gh,gd.N_gz)
    sav=zeros(gd.N_ghp,gd.N_gap,gd.N_gh,gd.N_gz)
    val_choice=zeros(gd.N_ghp,gd.N_gap,gd.N_gh,gd.N_gz)
    
    
    z_exo=zeros(gd.N_ghp,gd.N_ga,gd.N_gh,gd.N_gz)
    exp_val=zeros(gd.N_ga,)
    exp_dval=zeros(gd.N_ga,)
    
    W=zeros(gd.N_gap,gd.N_gh,gd.N_gz)
    Wa=zeros(gd.N_gap,gd.N_gh,gd.N_gz)
    V=zeros(gd.N_ga,gd.N_gh,gd.N_gz)
    Va=zeros(gd.N_ga,gd.N_gh,gd.N_gz)

    
    mask=zeros(Bool,gd.N_ga-1,)
    
    c_node=zeros(gd.N_ga,)
    z_node=zeros(gd.N_ga,)
    val_node=zeros(gd.N_ga,)
    ap_node=zeros(Int64,gd.N_ga,)

    c_tmp=zeros(gd.N_ga,)
    z_tmp=zeros(gd.N_ga,)

    c_out=zeros(gd.N_ga,)
    v_out=zeros(gd.N_ga,)
    

    V_old=zeros(gd.N_gap,gd.N_gh,gd.N_gz)
    segment=zeros(Int64,gd.N_ga,gd.N_gh,gd.N_gz)
    return DecisionMatricies(con,sav,val_choice,
                            z_exo,exp_val,exp_dval,
                            W,Wa,V,Va,
                            mask,
                            c_node,z_node,val_node,ap_node,
                            c_tmp,z_tmp,c_out,v_out,
                            V_old,segment)
end

struct Grids{T<:Union{Float32,Float64}}
    N_gap::Int64
    N_gh::Int64
    N_ghp::Int64
    N_gz::Int64
    N_ga::Int64
    agrid::Vector{T}
    apgrid::Vector{T}
    multgrid::Vector{T}
    hgrid::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}}
    zgrid::Vector{T}
    T_mat::Array{T,2}
end



function set_Grids(hhp)
    N_gap=200
    N_gh=7
    N_ghp=N_gh
    N_gz=7
    N_ga=200
    apgrid=exp.(exp.(range(log(log(hhp.amin+1.0)+1.0),stop=log(log(hhp.amax+1.0)+1.0),length=N_gap)).-1.0).-1.0
    agrid=exp.(exp.(range(log(log(hhp.amin+1.0)+1.0),stop=log(log(hhp.amax+1.0)+1.0),length=N_ga)).-1.0).-1.0
    multgrid=reverse(exp.(exp.(range(log(log(0.0+1.0)+1.0),stop=log(log(1.0+1.0)+1.0),length=N_gap)).-1.0).-1.0)
    #multgrid=reverse(collect(range(0.0,stop=1.0,length=N_gap)))
    hgrid=range(hhp.hmin,stop=hhp.hmax,length=N_ghp)
    #mc=rouwenhorst(N_gz,hhp.rho,hhp.sigma,0.0)
    mc=tauchen(N_gz,hhp.rho,hhp.sigma,0.0)
    mc_t=tauchen(N_gz,hhp.rho_t,hhp.sigma_t,0.0)
    zgrid=exp.(kron(mc.state_values,ones(N_gz,))+kron(ones(N_gz,),mc_t.state_values))
    T_mat=kron(mc.p,ones(N_gz,N_gz)).*kron(ones(N_gz,N_gz),mc_t.p)
    #println(sum(T_mat,dims=2))
    return Grids(N_gap,N_gh,N_ghp,N_gz^2,N_ga,agrid,apgrid,multgrid,hgrid,zgrid,T_mat)
end

function h_fun(h,hp)
    phi=0.06
    if (h==hp)
        return 0.0
    else
        return phi*hp
    end
end


function u_fun(c,hp,hhp)
    return hhp.theta*log(c)+(1.0-hhp.theta)*log(hp*hhp.kappa)
end


function u_fun(c,hp,theta,kappa)
    return theta*log(c)+(1.0-theta)*log(hp*kappa)
end
##################################################
# consumption from ee with and withoug binding constraint
##################################################

function get_c_from_ee(theta,∂W_a)
    return (∂W_a/theta)^(-1)
end

function get_c_from_ee(theta,∂W_a,λ)
    return (∂W_a*(1.0+λ)/theta)^(-1)
end



# sub_rnumderiv
function finite_diff!(dm,gd)
    dm.Va[1,:,:]=(dm.V[2,:,:]-dm.V[1,:,:])./(gd.apgrid[2]-gd.apgrid[1])
    dm.Va[2:end-1,:,:]=(dm.V[3:end,:,:]-dm.V[2:end-1,:,:])./(gd.apgrid[3:end]-gd.apgrid[2:end-1])
    dm.Va[end,:,:]=(dm.V[end,:,:]-dm.V[end-1,:,:])./(gd.apgrid[end]-gd.apgrid[end-1])
end

function init_V!(dm,gd,hhp)
    for j3 in 1:gd.N_gz, j1 in 1:gd.N_gap,j2 in 1:gd.N_gh 
        @inbounds dm.V[j1,j2,j3]=hhp.theta*log((gd.zgrid[j3]+hhp.r*gd.apgrid[j1]))+(1.0-hhp.theta)*log(hhp.kappa*gd.hgrid[j2])
    end
end

function update_Wa!(gd,dm,hhp)
    for j3 in 1:gd.N_gz, j2 in 1:gd.N_gh, j1 ∈ 1:gd.N_gap
        dm.Wa[j1,j2,j3]=hhp.beta*dot(dm.Va[j1,j2,:],gd.T_mat[j3,:])
    end
end

function update_W!(gd,dm,hhp)
    for j3 in 1:gd.N_gz, j2 in 1:gd.N_gh, j1 in 1:gd.N_gap
        dm.W[j1,j2,j3]=hhp.beta*dot(dm.V[j1,j2,:],gd.T_mat[j3,:])
    end
end


function sub_globalegm(dm,gd,hhp,hp)
    val=zeros(2,)
    idx_a_min=gd.N_ga
    idx_a_max=1
    dm.mask[:]=dm.exp_dval[2:end].>dm.exp_dval[1:end-1]
    if sum(dm.mask)>0
        exp_dval_up=maximum(dm.exp_dval[2:end][dm.mask])
        exp_dval_down=minimum(dm.exp_dval[1:end-1][dm.mask])
        if sum(dm.exp_dval.>exp_dval_up)==0
            idx_a_min=1            
        else
            # MINLOC
            a_min_tmp=minimum(dm.exp_dval[dm.exp_dval.>exp_dval_up])
            idx_a_min=findfirst(dm.exp_dval.==a_min_tmp)+1
        end
        if sum(dm.exp_dval.<exp_dval_down)==0
            idx_a_max=gd.N_ga
        else 
            #MAXLOC
            a_max_tmp=maximum(dm.exp_dval[dm.exp_dval.<exp_dval_down])
            idx_a_max=findfirst(dm.exp_dval.==a_max_tmp)-1
        end
    end
    for j1 in 1:gd.N_ga
        dm.c_tmp[j1]=get_c_from_ee(hhp.theta,dm.exp_dval[j1])
        dm.z_tmp[j1]=dm.c_tmp[j1]+gd.agrid[j1]

    end
    z_bar=dm.z_tmp[1]

    i_aprime=Int64.(range(1,stop=gd.N_ga,length=gd.N_ga))
    if idx_a_min<gd.N_ga && idx_a_max>1
        p=sortperm(dm.z_tmp[idx_a_min:idx_a_max])
        i_aprime[idx_a_min:idx_a_max]=i_aprime[p.+(idx_a_min-1)]
        dm.c_tmp[idx_a_min:idx_a_max]=dm.c_tmp[i_aprime[idx_a_min:idx_a_max]]
        dm.z_tmp[idx_a_min:idx_a_max]=dm.z_tmp[i_aprime[idx_a_min:idx_a_max]]
    end

    j1=0
    j2=0
    j3=0

    for j4 in 1:gd.N_ga
        if j4<idx_a_min || j4>idx_a_max
            j3+=1
            dm.c_node[j3]=dm.c_tmp[j4]
            dm.z_node[j3]=dm.z_tmp[j4]
            dm.ap_node[j3]=i_aprime[j4]
            j1=i_aprime[j4]
            j2=j4
            dm.val_node[j3]=u_fun(dm.c_node[j3],hp,hhp)+dm.exp_val[i_aprime[j4]]
        elseif j3>0 && i_aprime[j4]<j1
            continue
        else
            val[2]=u_fun(dm.z_tmp[j4]-gd.agrid[i_aprime[j4]],hp,hhp)+dm.exp_val[i_aprime[j4]]
            if j3==0
                j2=j4
            end
            j5=min(j4+10,idx_a_max)
            val[1]=maximum(u_fun.(max.(1.0e-7,dm.z_tmp[j4].-gd.agrid[i_aprime[j2:j5]]),hp,hhp.theta,hhp.kappa).+dm.exp_val[i_aprime[j2:j5]])
            if val[1]>val[2]
                continue
            end
            if dm.z_tmp[j4]<z_bar
                val[1]=u_fun(dm.z_tmp[j4]-gd.agrid[1],hp,hhp)+dm.exp_val[1]
                if val[1]>val[2]
                    continue
                end
            end
            j3+=1
            dm.c_node[j3]=dm.c_tmp[j4]
            dm.z_node[j3]=dm.z_tmp[j4]
            dm.ap_node[j3]=i_aprime[j4]
            j1=i_aprime[j4]
            j2=j4
            dm.val_node[j3]=u_fun(dm.c_node[j3],hp,hhp)+dm.exp_val[i_aprime[j4]]
        end

    end

    if dm.ap_node[1]>1
        ag=gd.agrid[1]
        b=gd.agrid[dm.ap_node[1]]
        delta_vp=dm.exp_val[dm.ap_node[1]]-dm.exp_val[1]
        sol=nlsolve((F,x)->sub_min_z!(F,x,ag,b,delta_vp,hp,hhp),(J,x)->J_sub_min_z!(J,x,ag,b,hhp),[dm.z_node[1]])
        dm.c_node[2:gd.N_ga]=dm.c_node[1:gd.N_ga-1]
        dm.z_node[2:gd.N_ga]=dm.z_node[1:gd.N_ga-1]
        dm.val_node[2:gd.N_ga]=dm.val_node[1:gd.N_ga-1]
        dm.c_node[1]=sol.zero[1]-1e-5
        dm.z_node[1]=dm.c_node[1]+gd.agrid[1]
        dm.val_node[1]=u_fun(dm.c_node[1],hp,hhp)+dm.exp_val[1]
        j3+=1
    end
    return j3
end


function sub_min_z!(F,x,ag,b,delta_vp,hp,hhp)
    pentalty=max(-x[1],0.0)*100.0
    c=max(x[1],1.0e-7+max(ag,b))
    F[1]=u_fun(c-ag,hp,hhp)-u_fun(c-b,hp,hhp)-delta_vp
    F[1]+=pentalty
end

function J_sub_min_z!(J,x,ag,b,hhp)
    J[1]=hhp.theta/(x[1]-ag)-hhp.theta/(x[1]-b)
end
#

#
function sub_egminterp(dm,gd,hhp,z_exo,hp,j3)
    for j1 in 1:gd.N_ga
        idx_node=binary_search_con(1,z_exo[j1],dm.z_node[1:j3],j3)
        u_1=u_fun(dm.c_node[1],hp,hhp)
        if idx_node==0
            if z_exo[j1]<=gd.agrid[1]
                dm.c_out[j1]=1.0e-7
                dm.v_out[j1]=-1.0e7
            else
                dm.c_out[j1]=z_exo[j1]-gd.agrid[1]
                dm.v_out[j1]=dm.val_node[1]-u_1+min(u_fun(dm.c_out[j1],hp,hhp),u_1)
            end
        else
            dm.c_out[j1]=interp_1d_given_index(dm.z_node[1:j3],z_exo[j1],dm.c_node[1:j3],idx_node)
            dm.v_out[j1]=interp_1d_given_index(dm.z_node[1:j3],z_exo[j1],dm.val_node[1:j3],idx_node)
        end
    end
end


function sub_valueiter()

    hhp=set_FirmParamsM()
    gd=set_Grids(hhp)
    dm=set_DecisionMatricies(gd)
    dm.val_choice[:].=-10e7
    init_V!(dm,gd,hhp)
    finite_diff!(dm,gd)

    set_z_exo!(dm,gd,hhp)
    dif=Inf
    maxitrs=1000
    i=0
    tol=1e-5
    j3m=0
    while dif>tol && i<maxitrs

        dm.val_choice[:].=-10e7
        update_Wa!(gd,dm,hhp)
        update_W!(gd,dm,hhp)
        for idx_y in 1:gd.N_gz
            for idx_hp in 1:gd.N_ghp
    
                hp=gd.hgrid[idx_hp]
                dm.exp_val[:]=dm.W[:,idx_hp,idx_y]
                dm.exp_dval[:]=dm.Wa[:,idx_hp,idx_y]
                j3m=sub_globalegm(dm,gd,hhp,hp)
    
    
                for idx_h in 1:gd.N_gh
                    sub_egminterp(dm,gd,hhp,dm.z_exo[idx_hp,:,idx_h,idx_y],hp,j3m)
                    dm.sav[idx_hp,:,idx_h,idx_y]=dm.z_exo[idx_hp,:,idx_h,idx_y]-dm.c_out
                    dm.con[idx_hp,:,idx_h,idx_y]=dm.c_out
                    dm.val_choice[idx_hp,:,idx_h,idx_y]=dm.v_out
                end
            end
            for idx_a in 1:gd.N_ga
                for idx_h in 1:gd.N_gh
                    idx_hp_max=argmax(dm.val_choice[:,idx_a,idx_h,idx_y])
                    dm.segment[idx_a,idx_h,idx_y]=idx_hp_max
                    dm.V[idx_a,idx_h,idx_y]=dm.val_choice[idx_hp_max,idx_a,idx_h,idx_y]
                    dm.Va[idx_a,idx_h,idx_y]=(1.0+hhp.r)*hhp.theta/(dm.con[idx_hp_max,idx_a,idx_h,idx_y])
                end
            end
        end
    

        dif=maximum(abs.((dm.V-dm.V_old))./maximum(abs.(dm.V_old)))
        dm.V_old[:,:,:]=dm.V
        if i%10==0
            @printf("Dif is %.10E \n",dif)
        end
        i+=1
    
    end
    return dm,hhp,gd
end


# fun_z
function set_z_exo!(dm,gd,hhp)
    for j4 ∈ 1:gd.N_gz, j2 ∈ 1:gd.N_ga,j3 ∈ 1:gd.N_gh, j1 ∈ 1:gd.N_ghp 
        @inbounds dm.z_exo[j1,j2,j3,j4]=gd.zgrid[j4]+(1.0+hhp.r)*gd.agrid[j2]-(1.0-hhp.zeta)*(gd.hgrid[j1]-gd.hgrid[j3])-h_fun(gd.hgrid[j3],gd.hgrid[j1])-hhp.r*(hhp.gamma*gd.zgrid[1]+hhp.zeta*gd.hgrid[j3])
    end
end


function sim_errors(dm,gd,hhp)
    

    T_cs=cumsum(gd.T_mat,dims=2)

    # initial conditions as in Fella
    y_idx=Int(floor(gd.N_gz/2))+1
    y_init=gd.zgrid[y_idx]
    a_idx=1
    h_idx=1
    a_init=gd.agrid[a_idx]
    T_sim=50000
    
    c=zeros(T_sim,)
    ap=zeros(T_sim,)
    hp=zeros(T_sim,)
    y=zeros(T_sim,)
    cp=zeros(gd.N_gz,T_sim)
    emuc=zeros(T_sim,)
    v_tmp=zeros(gd.N_gh,)
    ee_error=zeros(T_sim,)
    h_idx_vec=zeros(Int64,T_sim,)
    big_num=1000.0
    ap[1]=a_init
    y[1]=y_init
    hp[1]=gd.hgrid[h_idx]
    h_idx_vec[1]=h_idx

    for t in 1:T_sim-1
        
        for j1 in 1:gd.N_gh
            v_tmp[j1]=interp_1d(gd.agrid,ap[t],dm.val_choice[j1,:,h_idx,y_idx],gd.N_ga)
        end
        
        hp_idx=argmax(v_tmp)
        hp[t+1]=gd.hgrid[hp_idx]
        c[t]=interp_1d(gd.agrid,ap[t],dm.con[hp_idx,:,h_idx,y_idx],gd.N_ga)
        ap[t+1]=y[t]+(1.0+hhp.r)*ap[t]-(1.0-hhp.zeta)*(hp[t+1]-hp[t])-h_fun(hp[t],hp[t+1])-hhp.r*(hhp.gamma*gd.zgrid[1]+hhp.zeta*hp[t])-c[t]
        
        #c' for expectation
        for j1 in 1:gd.N_gz
            for j2 in 1:gd.N_gh
                v_tmp[j2]=interp_1d(gd.agrid,ap[t+1],dm.val_choice[j2,:,hp_idx,j1],gd.N_ga)
            end
            hp_exp_idx=argmax(v_tmp)
            cp[j1,t]=interp_1d(gd.agrid,ap[t+1],dm.con[hp_exp_idx,:,hp_idx,j1],gd.N_ga)
        end
        # compute emuc before updating y_idx
        emuc[t]=sum(gd.T_mat[y_idx,:].*(hhp.theta./cp[:,t]))*hhp.beta*(1.0+hhp.r)

        ee_error[t]= ap[t+1] >= gd.agrid[2] ? (hhp.theta/emuc[t])/(c[t])-1.0 : big_num
        y_rv=rand()
        y_idx=findfirst(T_cs[y_idx,:].>y_rv)
        y[t+1]=gd.zgrid[y_idx]
        h_idx=hp_idx

    end

    mean_assets=mean(ap[10001:end-1])
    mean_ee_error=mean(log10.(abs.(ee_error[10001:end-1][ee_error[10001:end-1].<big_num])))
    max_ee_error=maximum(log10.(abs.(ee_error[10001:end-1][ee_error[10001:end-1].<big_num])))

    open("sim_output.txt", "w") do file
        write(file, "Mean assets are "*string(mean_assets)*"\n")
        write(file, "Mean ee_errors are "*string(mean_ee_error)*"\n")
        write(file, "Max ee_errors are "*string(max_ee_error)*"\n")
    end

end

function main()
    dm,hhp,gd=sub_valueiter();
    sim_errors(dm,gd,hhp)
    @save "fella_2014_replication.bson" dm hhp gd

end

main()



