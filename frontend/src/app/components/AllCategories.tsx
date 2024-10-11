import Link from 'next/link';
import Image from 'next/image';
import { Category } from '../../../api/GroceriesApi';

export default function AllCategories({ categories }: { categories: Category[] }) {
    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {categories.map((category) => (
                <Link key={category.id} href={`/category/${category.id}`}>
                    <div className="card bg-base-200 shadow-xl border border-base-300 hover:bg-base-300 transition-colors duration-200">
                        <figure className="border-b border-base-300">
                            <Image
                                src={category.image}
                                alt={category.name}
                                width={400}
                                height={225}
                                className="w-full h-56 object-cover"
                            />
                        </figure>
                        <div className="card-body">
                            <h2 className="card-title text-base-content">{category.name}</h2>
                            <p className="text-base-content/70">{category.description}</p>
                            <div className="card-actions justify-end">
                                <button className="btn btn-primary">View</button>
                            </div>
                        </div>
                    </div>
                </Link>
            ))}
        </div>
    );
}