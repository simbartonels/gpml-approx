time_mean_cells = {};
value_mean_cells = {};
for j=1:size(x_value, 2)
    plot(x_value{j}, y_value{j}, '.', 'Color', plot_colors{method_id}, 'markersize', 3);
    if results.grad_norms{j}(end) < 1e-4
	%converged
	last_step_symbol = '+';
    elseif isnan(results.grad_norms{j}(end-1))
	%crashed
	last_step_symbol = 'x';
    else
	%timeout
	last_step_symbol = '*';
    end
    plot(x_value{j}(end), y_value{j}(end), last_step_symbol, 'Color', plot_colors{method_id}, 'markersize', 10);
    % plotting mean values
    initial_k = 2;
    if strcmp(method, 'Multiscale'), initial_k = 1; end
    for k=initial_k:max(size(x_value{j}, 2), size(time_mean_cells, 2))
        if k <= size(x_value{j}, 2)
            if k <= size(time_mean_cells, 2)
                time_mean_cells{k} = [time_mean_cells{k}, x_value{j}(k)];
                value_mean_cells{k} = [value_mean_cells{k}, y_value{j}(k)];
            else
%                 time_mean_cells{k} = x_value{j}(k);
                if k > initial_k
                    previous_cell = value_mean_cells{k-1}(1:end-1);
                    previous_cell2 = time_mean_cells{k-1}(1:end-1);
                else
                    previous_cell = [];
                    previous_cell2 = [];
                end
                value_mean_cells{k} = [previous_cell, y_value{j}(k)];
                time_mean_cells{k} = [previous_cell2, x_value{j}(k)];
            end
        else
            value_mean_cells{k} = [value_mean_cells{k}, y_value{j}(end)];
            time_mean_cells{k} = [time_mean_cells{k}, x_value{j}(end)];
        end
    end
end
plots{method_id} = plot(cellfun(@mean, time_mean_cells), cellfun(@mean, value_mean_cells), '-', 'Color', plot_colors{method_id}, 'linewidth', 1);
