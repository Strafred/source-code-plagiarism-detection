import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
            Scanner input = new Scanner(System.in);
            System.out.print("Enter the radius and length of a cylinder: "); 
            double radius = input.nextDouble();
            double length = input.nextDouble();
            double areas = radius * radius * 3.14159;
            double volumes = areas * length;
            System.out.println("The area is " + areas);
            System.out.println("The volume of the cylinder is " + volumes); }
}
