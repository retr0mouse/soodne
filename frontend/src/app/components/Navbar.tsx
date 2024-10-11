'use client'

import Link from 'next/link';
import ThemeToggle from './ThemeToggle';
import Image from 'next/image';
import ProfilePic from '../images/mouse.png';
import Cart from './Cart';

export default function Navbar() {

    return (
        <nav className="navbar bg-base-200">
            <div className="">
                <Link href="/" className="btn btn-ghost text-xl">Soodne</Link>
            </div>
            <div className="flex-1 justify-center">
                <div className="form-control w-full max-w-xl">
                    <input type="text" placeholder="Search" className="input input-bordered w-full" />
                </div>
            </div>
            <div className="flex-none">
                <ThemeToggle />
                <Cart />
                <div className="dropdown dropdown-end">
                    <label tabIndex={0} className="btn btn-ghost btn-circle avatar">
                        <div className="w-10 rounded-full">
                            <Image src={ProfilePic} alt="profile" width={40} height={40} />
                        </div>
                    </label>
                    <ul tabIndex={0} className="menu menu-sm dropdown-content mt-3 z-[1] p-2 shadow bg-base-100 rounded-box w-52">
                        <li><a>Profile</a></li>
                        <li><a>Settings</a></li>
                        <li><a>Logout</a></li>
                    </ul>
                </div>
            </div>
        </nav>
    );
}
