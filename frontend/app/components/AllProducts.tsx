'use client'
import { Product } from "../../api/GroceriesApi";
import { useCart } from '../context/CartContext';

export default function AllProducts({ products }: { products: Product[] }) {
    const { addToCart } = useCart();

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {products.map((product, index) => (
            <div key={index} className="card bg-base-200 shadow-xl">
              <figure><img src={product.image} alt={product.title} /></figure>
              <div className="card-body">
                <h2 className="card-title">{product.title}</h2>
                <p>{product.description}</p>
                <div className="flex justify-between items-center mt-4">
                  <span className="text-lg font-bold">${product.price.toFixed(2)}</span>
                  <span className="text-sm">${product.price_per_unit.toFixed(2)} / {product.unit_of_measure}</span>
                </div>
                <div className="card-actions justify-end mt-4">
                  <button className="btn btn-primary" onClick={() => addToCart(product)}>Add to cart</button>
                </div>
              </div>
            </div>
          ))}
        </div>
    );
}
