import java.util.Scanner;
public class Level3 
{
    public static void main(String[] args) 
    {
        Scanner x = new Scanner(System.in);
        double vol = 0;
        double luasAlas = 0;
        System.out.print("Enter the radius and length of a cylinder: ");
        double radius = x.nextDouble();
        double length = x.nextDouble();
        luasAlas = radius * radius * 3.14159;
        vol = luasAlas * length;
        System.out.println("The area is " + luasAlas);
        System.out.println("The volume of the cylinder is " + vol);
    }
}
