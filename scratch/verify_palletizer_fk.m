% Script completo: Comparación FK (DH vs Geométrica) e IK para Robot Agrícola (CR4)
clear; clc;

% =========================================================================
% 1. PARÁMETROS DEL ROBOT AGRÍCOLA (Distancias en metros)
% =========================================================================
Oz    = 0.319;   % Altura del pivote O desde la base A
L_OC  = 0.540;   % Longitud del brazo inferior (Lower Arm)
L_CH  = 0.380;   % Longitud de la extensión superior (Extension)
L_HEE = 0.060;   % Extensión horizontal de la brida (Tool Extension)
L_TCP = 0.000;   % Offset vertical de la herramienta (Tool Offset)

% =========================================================================
% 2. PUNTOS DE VERIFICACIÓN (Medidos de la simulación)
% Cada fila: [q1, q2, q3, q4] (grados) | [x_ref, y_ref, z_ref] (metros)
% =========================================================================
test_cases = {
    'Home',  [0, 0, 0, 0],             [0.4400, 0.0000, 0.8590];
    'Pos1',  [0, 25, 0, 0],            [0.6682, 0.0000, 0.8084];
    'Pos2',  [0, 25, 53.7, 0],         [0.5132, 0.0000, 0.5022];
    'Pos3',  [-54.5, 17.4, -2.9, 0],   [0.3490, -0.4893, 0.8535];
    'Pos4',  [35.3, -13.1, 9, 0],      [0.2554, 0.1808, 0.7855]
};

fprintf('=================================================================\n');
fprintf('       VALIDACIÓN CINEMÁTICA: ROBOT AGRÍCOLA (CR4 TEMPLATE)\n');
fprintf('=================================================================\n\n');

for i = 1:size(test_cases, 1)
    name = test_cases{i, 1};
    q = test_cases{i, 2};
    p_ref = test_cases{i, 3};
    
    q1 = q(1); q2 = q(2); q3 = q(3); q4 = q(4);
    
    fprintf('>>> CASO: %s | q = [%0.1f, %0.1f, %0.1f, %0.1f] (grados)\n', ...
        name, q1, q2, q3, q4);
    
    %% 3. FK MEDIANTE DENAVIT-HARTENBERG (DH)
    % Mapeo de ángulos de articulación físicos a ángulos DH relativos:
    q2_dh = q2 - 90;
    q3_dh = q3 - q2 + 90;
    q4_dh = 180 - q3;
    q5_dh = q4;  % El giro final del plato
    
    % Matrices de transformación eslabón a eslabón
    A_10 = matriz_A(q1, Oz, 0, -90);
    A_21 = matriz_A(q2_dh, 0, L_OC, 0);
    A_32 = matriz_A(q3_dh, 0, L_CH, 0);
    A_43 = matriz_A(q4_dh, 0, -L_HEE, 90);
    A_54 = matriz_A(q5_dh, L_TCP, 0, 0);
    
    T = A_10 * A_21 * A_32 * A_43 * A_54;
    px_dh = T(1,4); py_dh = T(2,4); pz_dh = T(3,4);
    
    %% 4. FK MEDIANTE MÉTODO GEOMÉTRICO (2D Plano r-z)
    % Coordenadas del codo C y extremo H en el plano vertical
    r_c = L_OC * sind(q2);
    z_c = Oz + L_OC * cosd(q2);
    
    r_h = r_c + L_CH * cosd(q3);
    z_h = z_c - L_CH * sind(q3);
    
    % Extensión final del TCP aplicando offsets rígidos de muñeca
    r_tcp = r_h + L_HEE;
    z_tcp = z_h - L_TCP;
    
    % Proyección al espacio 3D final por la rotación de la base q1
    px_geom = r_tcp * cosd(q1);
    py_geom = r_tcp * sind(q1);
    pz_geom = z_tcp;
    
    %% 5. CINEMÁTICA INVERSA ANALÍTICA (IK)
    % Usamos la posición del TCP calculada por DH para verificar reversibilidad
    x = px_dh; y = py_dh; z = pz_dh;
    
    % q1: Ángulo de la base
    ik_q1 = atan2d(y, x);
    
    % Desplazamiento inverso del TCP al punto H en el plano r-z
    r_target = sqrt(x^2 + y^2);
    r_prime = r_target - L_HEE;
    z_prime = z + L_TCP - Oz;
    
    % Distancia euclídea del punto H respecto al pivote O en el plano r-z
    D2 = r_prime^2 + z_prime^2;
    D = sqrt(D2);
    
    % Triángulo O-C-H: Teorema del coseno
    % Ángulo interno gamma en el codo C:
    cos_gamma = (L_OC^2 + L_CH^2 - D2) / (2 * L_OC * L_CH);
    gamma = acosd(max(-1, min(1, cos_gamma)));
    
    % Ángulo interno beta en el pivote O:
    cos_beta = (L_OC^2 + D2 - L_CH^2) / (2 * L_OC * D);
    beta = acosd(max(-1, min(1, cos_beta)));
    
    % Ángulo del vector O-H con la horizontal:
    theta = atan2d(z_prime, r_prime);
    
    % Articulaciones físicas solucionadas (Codo Arriba):
    ik_q2 = 90 - theta - beta;
    ik_q3 = ik_q2 + 90 - gamma;
    ik_q4 = q4; % Ángulo yaw del plato copiado directamente
    
    %% 6. COMPARACIÓN DE RESULTADOS
    fprintf('  [Ref]   Pos: X = %0.4f, Y = %0.4f, Z = %0.4f\n', p_ref(1), p_ref(2), p_ref(3));
    fprintf('  [DH]    Pos: X = %0.4f, Y = %0.4f, Z = %0.4f (Error RMS: %0.2e)\n', ...
        px_dh, py_dh, pz_dh, norm([px_dh, py_dh, pz_dh] - p_ref));
    fprintf('  [Geom]  Pos: X = %0.4f, Y = %0.4f, Z = %0.4f (Error RMS: %0.2e)\n', ...
        px_geom, py_geom, pz_geom, norm([px_geom, py_geom, pz_geom] - p_ref));
    fprintf('  [IK]    q_calc = [%0.2f, %0.2f, %0.2f, %0.2f] (Error: [%0.2e, %0.2e, %0.2e])\n\n', ...
        ik_q1, ik_q2, ik_q3, ik_q4, ik_q1-q1, ik_q2-q2, ik_q3-q3);
end

% =========================================================================
% FUNCIONES AUXILIARES
% =========================================================================
function A_ji = matriz_A(th, d, a, af)
    % Genera la matriz de transformación DH estándar
    A_ji = [cosd(th) -cosd(af)*sind(th)  sind(af)*sind(th) a*cosd(th);
            sind(th)  cosd(af)*cosd(th) -sind(af)*cosd(th) a*sind(th);
            0         sind(af)           cosd(af)          d;
            0         0                  0                 1];
end
