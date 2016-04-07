function weights = g_u(I, u, alpha, beta)

ind = sub2ind(size(I), u(2,:), u(1,:));

[gx, gy] = derivative5(I, 'x', 'y');

weights = exp(-alpha*(sqrt( gx(ind).^2 + gy(ind).^2 )).^beta);

end