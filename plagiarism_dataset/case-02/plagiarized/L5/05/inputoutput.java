import java.util.*;
class inputoutput{
	static double rad,panjang;
	static void input()
	{
		Scanner masukan = new Scanner(System.in);
		rad = masukan.nextDouble();
		panjang = masukan.nextDouble();
	}
	static double hitArea()
	{
		return rad * rad * 3.14159;
	}
	public static void main(String[] args){
		System.out.print("Enter the radius and length of a cylinder: ");
		input();
		double vol, a;
		a = hitArea();
		vol = a * panjang;
		System.out.println("The area is " + a);
		System.out.println("The volume of the cylinder is " + vol);		
	}
}