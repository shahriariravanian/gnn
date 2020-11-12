include("gnn.jl")
include("model.jl")
include("prepare.jl")
include("train.jl")
include("neural_ode.jl")

const models_dir="models_enhanced"

function generate_all(; mix=false)
    if !ispath(models_dir)
        mkdir(models_dir)
    end

    ℳ = TenTusscherModel()

    names = ["longqt", "shortqt", "ito"]
    # mods = [[(:g_Kr, 0.5)], [(:g_CaL, 0.5)], [(:g_to, 2.0)]]
    mods = [[(:g_Kr, 0.25)], [(:g_CaL, 0.25)], [(:g_to, 3.0)]]

    for (name, mod) in zip(names, mods)
        @info "generating $name\n"
        ℬ = generate_model(ℳ, name, mod, mix)
        generate_submodels(ℬ, name; ηs=1f-4:1f-4:2f-3)        
    end
end

function generate_model(ℳ, name, mod, mix)
    path = model_path(models_dir, name)
    if !ispath(path)
        ℳ1 = modify_model(ℳ, mod)
        ℬ = prepare(ℳ, ℳ1; mix=mix)
        ℬ = train(ℬ)
        save_model(ℬ, path)
    else
        ℬ = load_model(path)
    end
    return ℬ
end


function generate_submodels(ℬ, name; ηs=1f-4:1f-4:2f-3)
    for η in ηs
        @info "\ttraining for $η\n"
        path = submodel_path(models_dir, name, η)
        if !ispath(path)
            ℛ = retrain(ℬ, η)
            save_model(ℛ, path)
        end
    end
end
