/*
    Bicubic surface with Three.js

    Created by Rexirl on 2021-06-19.
    The bicubic interpolation is an implementation of matrix math as described at https://en.wikipedia.org/wiki/Bicubic_interpolation .
    Licensed under the MIT license.
 */


const makePoints = () => {
    const scale = 10;
    let points = [];
    const w = 4;
    for (let ix = 0; ix < w; ++ix)
    for (let iy = 0; iy < w; ++iy) {
        const u = ix / (w - 1);
        const v = iy / (w - 1);

        // generate the point location
        // - to do: a moderately interesting shape
        // (the current code is arbitrary)
        let x, y, z;
        x = u * 2 - 1;
        y = v * 2 - 1;
        z = Math.max(0, 1 - (x*x + y*y));
        z = Math.max(0, 1 - (x*x + y*(y + 0.5 * x)));
        z -= 0.75;
        x += z * (y + 0.25 * x);

        points.push(x * scale, y * scale, z * scale);
    }
    return points;
};


// assume points contains 4*4*3 values (16 positions)
// convenience function constructing a THREE.Vector3 from input point data
const getPoint = (points, ix, iy) => {
    const w = 4;
    ix = Math.min(Math.max(ix, 0), w - 1);
    iy = Math.min(Math.max(iy, 0), w - 1);
    const i0 = (iy * w + ix) * 3;
    return new THREE.Vector3(points[i0 + 0], points[i0 + 1], points[i0 + 2]);
};

const interpolateLinearly = (points, u, v) => {
    const transform = (x) => {
        return (x + 1) / 3;
    };
    u = transform(u);
    v = transform(v);

    const w = 4;
    const uu = u * (w - 1);
    const vv = v * (w - 1);
    const u0 = Math.floor(uu);
    const v0 = Math.floor(vv);
    const p00 = getPoint(points, u0, v0);
    const p10 = getPoint(points, u0 + 1, v0);
    const p01 = getPoint(points, u0, v0 + 1);
    const p11 = getPoint(points, u0 + 1, v0 + 1);
    return p00.lerp(p10, uu - u0).lerp(p01.lerp(p11, uu - u0), vv - v0);
};



//
// The part that is relevant to bicubic interpolation.
//

const computeBicubicInterpolationMatrix = (points) => {
    const w = 4;
    const point = (ix, iy) => { return getPoint(points, ix + 1, iy + 1); };


    // partial derivative with respect to x
    const dx = (ix, iy) => {
        // return (points[ix + 1][iy] - points[ix - 1][iy]) / 2.0;
        return point(ix + 1, iy).sub(point(ix - 1, iy)).divideScalar(2.0);
    };

    // partial derivative with respect to y
    const dy = (ix, iy) => {
        // return (points[ix][iy + 1] - points[ix][iy - 1]) / 2.0;
        return point(ix, iy + 1).sub(point(ix, iy - 1)).divideScalar(2.0);
    };

    // numeric cross partial derivative
    // (partial derivatives with respect to x of partial derivatives with respect to y,
    // or (equivalently) "reversed order" (y then x))
    const dxy = (ix, iy) => {
        // return (dx(ix, iy + 1) - dx(ix, iy - 1)) / 2.0;
        return dx(ix, iy + 1).sub(dx(ix, iy - 1)).divideScalar(2.0);
    };
    // - to do: check correctness

    // the four "central" values (bounding the region to be interpolated)
    const f00 = point(0, 0);
    const f10 = point(1, 0);
    const f01 = point(0, 1);
    const f11 = point(1, 1);

    // numeric partial derivatives with respect to x
    const fx00 = dx(0, 0);
    const fx10 = dx(1, 0);
    const fx01 = dx(0, 1);
    const fx11 = dx(1, 1);

    // numeric partial derivatives with respect to y
    const fy00 = dy(0, 0);
    const fy10 = dy(1, 0);
    const fy01 = dy(0, 1);
    const fy11 = dy(1, 1);

    // numeric cross partial derivatives
    const fxy00 = dxy(0, 0);
    const fxy10 = dxy(1, 0);
    const fxy01 = dxy(0, 1);
    const fxy11 = dxy(1, 1);

    const fmat = [
        f00,  f01,  fy00,  fy01,
        f10,  f11,  fy10,  fy11,
        fx00, fx01, fxy00, fxy01,
        fx10, fx11, fxy10, fxy11
    ];

    const A = new THREE.Matrix4();
    A.set( 1,  0,  0,  0,
           0,  0,  1,  0,
          -3,  3, -2, -1,
           2, -2,  1,  1);
    const Ainv = A.clone().transpose();

    // = s * v
    const multiplyScalarMat4AndVec3Mat4 = (s, v) => {
        // compute a column-major 4x4 matrix with 3D vector entries
        const r = [];
        for (let x = 0; x < 4; ++x) // over columns in r
        for (let y = 0; y < 4; ++y) { // over rows in r
            const p = new THREE.Vector3();
            for (let i = 0; i < 4; ++i) { // over columns in s and rows in v
                p.add(v[x * 4 + i].clone().multiplyScalar(s[i * 4 + y]));
            }
            r.push(p);
        }
        return r;
    };

    // = v * s
    const multiplyVec3Mat4AndScalarMat4 = (v, s) => {
        // compute a column-major 4x4 matrix with 3D vector entries
        const r = [];
        for (let x = 0; x < 4; ++x) // over columns in r
        for (let y = 0; y < 4; ++y) { // over rows in r
            const p = new THREE.Vector3();
            for (let i = 0; i < 4; ++i) { // over columns in v and rows in s
                p.add(v[i * 4 + y].clone().multiplyScalar(s[x * 4 + i]));
            }
            r.push(p);
        }
        return r;
    };

    let a = multiplyScalarMat4AndVec3Mat4(A.elements, fmat);
    a = multiplyVec3Mat4AndScalarMat4(a, Ainv.elements); // multiply with transpose(A) to get the final matrix
    return a;
};

// a: matrix from computeBicubicInterpolationMatrix
const interpolateBicubically = (a, u, v) => {
    // Now for the part that depends on u and v (which means it cannot be precomputed).
    // Conceptually, we now compute U * a * V, where
    const U = [1, u, u*u, u*u*u]; // (considered a 1x4 matrix) and
    const V = [1, v, v*v, v*v*v]; // (considered a 4x1 matrix).
    const Ua = [];
    for (let i = 0; i < 4; ++i) { // over columns in U and a
        Ua[i] = new THREE.Vector3();
        for (let j = 0; j < 4; ++j) { // over rows in a
            Ua[i].add(a[i * 4 + j].clone().multiplyScalar(U[j]));
        }
    }
    const p = new THREE.Vector3();
    for (let i = 0; i < 4; ++i) { // over columns in Ua and rows in V
        p.add(Ua[i].multiplyScalar(V[i]));
    }
    return p;
};

const interpolate = (mode, pointsOrMatrix, u, v) => {
    switch (mode) {
        case 'linear':
            return interpolateLinearly(pointsOrMatrix, u, v);
        case 'bicubic':
            return interpolateBicubically(pointsOrMatrix, u, v);
        default:
            console.log('unrecognized interpolation mode "' + mode + '"');
            return new THREE.Vector3();
    }
};



//
// Everything past this point is essentially just boilerplate (setup and display).
//

const init = () => {

    const options = {
        interpolationMode: 'bicubic',
        segments: 10
    };

    //

    const renderer = new THREE.WebGLRenderer( { antialias: true } );
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    //

    const camera = new THREE.PerspectiveCamera( 27, window.innerWidth / window.innerHeight, 1, 3500 );
    //camera.position.z = 64;

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    camera.position.set(0, 25, 10);
    controls.update();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color( 0x050505 );

    // Lighting.
    {
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0xffffff, 0.6);
        hemiLight.color.setHSL(0.6, 1, 0.6);
        hemiLight.groundColor.setHSL(0.095, 1, 0.75);
        hemiLight.position.set(0, 50, 0);
        scene.add(hemiLight);

        const pointLight = new THREE.PointLight(0xffffff, 1, 100);
        pointLight.position.set(20, 20, 20);
        scene.add(pointLight);
    }

    const material = new THREE.MeshLambertMaterial( {
        side: THREE.DoubleSide,
        vertexColors: true,
        wireframe: false
    } );


    const gui = new dat.GUI();
    gui.add(material, 'wireframe');
    gui.add(options, 'interpolationMode', ['linear', 'bicubic']);
    gui.add(options, 'segments', 1, 40).step(1);


    window.addEventListener( 'resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize( window.innerWidth, window.innerHeight );
    });

    let mesh = null;
    const createMesh = () => {
        if (mesh) {
            mesh.geometry.dispose();
            scene.remove(mesh);
        }

        const mode = options.interpolationMode;
        const isBicubic = mode == 'bicubic';

        const geometry = new THREE.BufferGeometry();

        const indices = [];
    
        const vertices = [];
        const normals = [];
        const colors = [];
    
        const size = 20;
    
        // generate vertices, normals and color data for a simple grid geometry
        const points = makePoints();
        const bicubicInterpolationMatrix = computeBicubicInterpolationMatrix(points);
        const pointsOrMatrix = !isBicubic ? points : bicubicInterpolationMatrix;
        const segments = options.segments;
        for (let i = 0; i <= segments; ++i) {
            const v = i / segments;
            const y = (v - 0.5) * size;
            for (let j = 0; j <= segments; ++j) {
                const u = j / segments;
                const x = (u - 0.5) * size;
    
                const computeVertex = (u, v) => {
                    return interpolate(mode, pointsOrMatrix, u, v);
                };
                const vertex = computeVertex(u, v);
                const dvdx = computeVertex(u + 1 / segments, v).sub(computeVertex(u - 1 / segments, v)).divideScalar(2.0);
                const dvdy = computeVertex(u, v + 1 / segments).sub(computeVertex(u, v - 1 / segments)).divideScalar(2.0);
                const normal = dvdx.clone().cross(dvdy).negate();
                vertices.push(vertex.x, vertex.y, vertex.z);
                normals.push(normal.x, normal.y, normal.z);
    
                const r = (x / size) + 0.5;
                const g = (y / size) + 0.5;
    
                colors.push( r, g, 1 );
            }
        }
    
        // generate indices (data for element array buffer)
    
        for (let i = 0; i < segments; ++i)
        for (let j = 0; j < segments; ++j) {
    
            const a = i * ( segments + 1 ) + ( j + 1 );
            const b = i * ( segments + 1 ) + j;
            const c = ( i + 1 ) * ( segments + 1 ) + j;
            const d = ( i + 1 ) * ( segments + 1 ) + ( j + 1 );
    
            // create a quad (of two triangles)
            indices.push(a, b, d);
            indices.push(b, c, d);
        }
    
        //
    
        geometry.setIndex( indices );
        geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( vertices, 3 ) );
        geometry.setAttribute( 'normal', new THREE.Float32BufferAttribute( normals, 3 ) );
        geometry.setAttribute( 'color', new THREE.Float32BufferAttribute( colors, 3 ) );

        mesh = new THREE.Mesh( geometry, material );
        mesh.rotation.x = -Math.PI / 2;
        controls.target = mesh.position;
        scene.add(mesh);
    };

    let oldInterpolationMode, oldSegments;
    const render = () => {
        const time = Date.now() * 0.001;
        /*mesh.rotation.x = time * 0.25;
        mesh.rotation.y = time * 0.5;*/
        if (oldInterpolationMode != options.interpolationMode ||
            oldSegments != options.segments)
        {
            oldInterpolationMode = options.interpolationMode;
            oldSegments = options.segments;
            createMesh();
        }
        renderer.render( scene, camera );
    };

    const animate = () => {
        requestAnimationFrame( animate );
        render();
        //stats.update();
    };
    animate();
};

init();
