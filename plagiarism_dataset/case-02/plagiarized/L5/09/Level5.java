import java.util.Scanner;
public class Level5 
{
    public static void main(String[] args) 
    {
        Scanner input = new Scanner(System.in);
        double radius, tinggi;
        System.out.print("Enter the radius and length of a cylinder: ");
        radius = input.nextDouble();
        tinggi = input.nextDouble();
        hitung(radius, tinggi);
    }
    private static void hitung(double radius, double tinggi) 
    {
        double area = radius * radius * 3.14159;
        double volume = radius * radius * 3.14159 * tinggi;
        System.out.println("The area is " + area);
        System.out.println("The volume of the cylinder is " + volume);
    }
}
