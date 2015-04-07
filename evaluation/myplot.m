time_mean_cells = {};
value_mean_cells = {};
for j=1:size(x_value, 2)
    plot(x_value{j}, y_value{j}, '.', 'Color', plot_colors{method_id});
    for k=1:max(size(x_value{j}, 2), size(time_mean_cells, 2))
        make_copy_from_last_cell = true;
        if k <= size(x_value{j}, 2)
            if size(time_mean_cells, 2) >= k
                time_mean_cells{k} = [time_mean_cells{k}, x_value{j}(k)];
                value_mean_cells{k} = [value_mean_cells{k}, y_value{j}(k)];
            else
                time_mean_cells{k} = x_value{j}(k);
                
                if k > 1 
                    previous_cell = value_mean_cells{k-1}(1:end-1);
                else
                    previous_cell = [];
                end
                value_mean_cells{k} = [previous_cell, y_value{j}(k)];
            end
        else
            value_mean_cells{k} = [value_mean_cells{k}, y_value{j}(end)];
        end
    end
end
plots{method_id} = plot(cellfun(@mean, time_mean_cells), cellfun(@mean, value_mean_cells), '-', 'Color', plot_colors{method_id});