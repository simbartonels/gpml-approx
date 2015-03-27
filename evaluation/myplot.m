time_mean_cells = {};
value_mean_cells = {};
for j=1:size(x_value, 2)
    plot(x_value{j}, y_value{j}, 'x', 'Color', plot_colors{method_id});
    for k=1:size(x_value{j}, 2)
        if size(time_mean_cells, 2) >= k
            time_mean_cells{k} = [time_mean_cells{k}, x_value{j}(k)];
            value_mean_cells{k} = [value_mean_cells{k}, y_value{j}(k)];
        else 
            time_mean_cells{k} = x_value{j}(k);
            value_mean_cells{k} = y_value{j}(k);
        end
    end
end
plots{method_id} = plot(cellfun(@mean, time_mean_cells), cellfun(@mean, value_mean_cells), '-', 'Color', plot_colors{method_id});