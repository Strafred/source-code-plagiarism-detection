import java.util.*;
class inputoutput{
	public static void main(String[] args){
		Scanner masukan = new Scanner(System.in);
		System.out.print("Enter the radius and length of a cylinder: ");
		double rad = masukan.nextDouble();
		double panjang = masukan.nextDouble();
		double a = rad * rad * 3.14159;
		double vol = a * panjang;
		System.out.println("The area is " + a);
		System.out.println("The volume of the cylinder is " + vol);		
	}
}