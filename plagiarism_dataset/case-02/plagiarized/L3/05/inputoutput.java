import java.util.*;
class inputoutput{
	public static void main(String[] args){
		Scanner masukan = new Scanner(System.in);
		double rad,panjang;
		System.out.print("Enter the radius and length of a cylinder: ");
		rad = masukan.nextDouble();
		panjang = masukan.nextDouble();
		double a;
		a = rad * rad * 3.14159;
		double vol;
		vol = a * panjang;
		System.out.println("The area is " + a);
		System.out.println("The volume of the cylinder is " + vol);		
	}
}