'use client'
import {useCart} from '../context/CartContext';
import {Product} from "../../types/ProductTypes";
import Image, {StaticImageData} from "next/image";
import RimiLogo from "../images/rimi_logo.png";
import MaximaLogo from "../images/maxima_logo.png";

export default function AllProducts({products}: { products: Product[] }) {
    const storeLogos = new Map<number, StaticImageData>([
        [1, RimiLogo],
        [2, MaximaLogo]
    ]);

    const {addToCart} = useCart();

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {products.map((product, index) => (
                <div key={index} className="card bg-base-200 drop-shadow">
                    <figure className={'h-56 bg-white py-0'}>
                        <Image width={500} height={500} className={'block h-full w-auto max-w-full max-h-full'}
                               src={product.store_data[0].store_image_url ?? ""}
                               alt={product.name}/>
                    </figure>
                    <div className="card-body">
                        <h2 className="card-title">{product.name}</h2>
                        <p>{product.description}</p>
                        <div className="flex justify-between items-center mt-4">
                            {product.store_data.map((data) => {
                                return (
                                    <div className={"flex gap-2 items-center"}>
                                        <figure className={'h-8 rounded-md'}>
                                            <Image
                                                className={'block h-full w-auto max-w-full max-h-full'}
                                                width={50}
                                                height={50}
                                                src={storeLogos.get(data.store_id)}
                                                alt={'rimi logo'}
                                            />
                                        </figure>
                                        <span className="text-lg font-bold">€{data.price}</span>
                                    </div>
                                );
                            })}
                        </div>
                        <div className="card-actions justify-center mt-4 w-full">
                            <button className="btn btn-primary w-full" onClick={() => addToCart(product)}>Add to cart</button>
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
}
